"""
IMOU Camera Integration Module - FIXED SIGNATURE
Исправленная версия с правильной подписью
"""

import cv2
import requests
import hashlib
import time
import uuid
import logging
import socket
import struct
from typing import Optional, Dict
from datetime import datetime
from face_app.core.config.env import (
    IMOU_APP_ID, IMOU_APP_SECRET, IMOU_DEVICE_ID, 
    IMOU_CHANNEL_ID, IMOU_REGION
)

logger = logging.getLogger(__name__)


class ImouCamera:
    """Класс для работы с IMOU камерой через OpenAPI"""
    
    API_REGIONS = {
        'cn': 'https://openapi-cn.easy4ip.com',
        'sg': 'https://openapi-sg.easy4ip.com',
        'eu': 'https://openapi-eu.easy4ip.com',
        'us': 'https://openapi-us.easy4ip.com',
        'ap': 'https://openapi-ap.easy4ip.com'
    }
    
    
    # Class-level cache to persist across instances
    _token_cache = {
        "access_token": None,
        "expire_time": 0,
        "stream_url": None,
        "live_token": None
    }

    def __init__(self):
        self.app_id = IMOU_APP_ID
        self.app_secret = IMOU_APP_SECRET
        self.device_id = IMOU_DEVICE_ID
        self.channel_id = str(IMOU_CHANNEL_ID)
        self.region = IMOU_REGION
        
        if self.region not in self.API_REGIONS:
            logger.warning(f"Region '{self.region}' unknown, using 'sg'")
            self.region = 'sg'
        
        self.base_url = self.API_REGIONS[self.region]
        self.base_url = self.API_REGIONS[self.region]
        
        # Load from cache
        self.token = ImouCamera._token_cache["access_token"]
        self.token_expire_time = ImouCamera._token_cache["expire_time"]
        self.live_token = ImouCamera._token_cache["live_token"]
        # Do not load stream_url blindly, as it might be wrong quality
        self.stream_url = None
        
        logger.info(f"IMOU Camera initialized: region={self.region}")
        logger.info(f"API URL: {self.base_url}")
    
    def _check_system_time(self):
        """Проверка синхронизации системного времени с NTP"""
        try:
            NTP_SERVER = "pool.ntp.org"
            TIME1970 = 2208988800
            
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.settimeout(3)
            data = b'\x1b' + 47 * b'\0'
            client.sendto(data, (NTP_SERVER, 123))
            data, _ = client.recvfrom(1024)
            
            if data:
                t = struct.unpack('!12I', data)[10]
                t -= TIME1970
                ntp_time = datetime.fromtimestamp(t)
                system_time = datetime.now()
                diff = abs((ntp_time - system_time).total_seconds())
                
                logger.info(f"NTP time: {ntp_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"System time: {system_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Time difference: {diff:.2f} seconds")
                
                if diff > 5:
                    logger.warning(f"⚠ System clock off by {diff:.2f}s! Please sync!")
                    return False
                return True
        except Exception as e:
            logger.warning(f"NTP check failed: {e}")
            return True
    
    def _generate_sign(self, params: Dict = None) -> Dict:
        """
        Генерация подписи по ПРАВИЛЬНОЙ документации IMOU
        
        Правильный формат строки подписи:
        time:{time},nonce:{nonce},appSecret:{appSecret}
        
        БЕЗ params, БЕЗ appId, БЕЗ ver!
        """
        current_time_s = int(time.time())
        nonce = str(uuid.uuid4())
        
        sign_string = f"time:{current_time_s},nonce:{nonce},appSecret:{self.app_secret}"
        
        logger.debug(f"Time: {current_time_s}")
        logger.debug(f"Nonce: {nonce}")
        logger.debug(f"Sign string: {sign_string}")
        
        sign = hashlib.md5(sign_string.encode('utf-8')).hexdigest()
        
        logger.debug(f"Generated sign: {sign}")
        
        return {
            "sign": sign,
            "time": current_time_s,
            "nonce": nonce
        }
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Выполнить запрос к IMOU API"""
        try:
            sign_data = self._generate_sign(params)
            
            payload = {
                "system": {
                    "ver": "1.0",
                    "appId": self.app_id,
                    "sign": sign_data["sign"],
                    "time": sign_data["time"],
                    "nonce": sign_data["nonce"]
                },
                "id": str(uuid.uuid4()),
                "params": params
            }
            
            url = f"{self.base_url}{endpoint}"
            
            logger.debug(f"Request to: {url}")
            logger.debug(f"Payload: {payload}")
            
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            logger.debug(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Response data: {data}")
                
                result = data.get('result', {})
                code = result.get('code')
                
                if code == '0':
                    return result.get('data')
                elif code == 'LV1001':
                    # "The video live exists" - это нормально, просто поток уже создан
                    logger.info(f"API notice [{code}]: The video live exists (already bound)")
                    return None
                else:
                    error_msg = result.get('msg', 'Unknown error')
                    logger.error(f"API error [{code}]: {error_msg}")
                    return None
            else:
                logger.error(f"HTTP error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Request error: {e}", exc_info=True)
            return None
    
    def get_access_token(self) -> bool:
        """Получить токен доступа"""
        try:
            if self.token and time.time() < self.token_expire_time - 300:
                logger.debug("Using cached token")
                return True
            
            logger.info("Requesting new access token...")
            
            self._check_system_time()
            
            data = self._make_request("/openapi/accessToken", {})
            
            if data:
                self.token = data.get('accessToken')
                expire_time = data.get('expireTime', 7200)
                self.token_expire_time = time.time() + expire_time
                
                # Update Cache
                ImouCamera._token_cache["access_token"] = self.token
                ImouCamera._token_cache["expire_time"] = self.token_expire_time
                
                logger.info(f"✓ Access token obtained (expires in {expire_time}s)")
                logger.debug(f"Token: {self.token[:20]}...")
                return True
            else:
                logger.error("✗ Failed to get access token")
                return False
                
        except Exception as e:
            logger.error(f"Token error: {e}", exc_info=True)
            return False
    
    def bind_device_live(self, stream_id: int = 1) -> bool:
        """
        Создать привязку устройства для live-трансляции
        stream_id: 0 = HD main stream, 1 = SD sub stream
        """
        try:
            if not self.get_access_token():
                return False
            
            logger.info(f"Binding device for live streaming (streamId={stream_id})...")
            
            data = self._make_request(
                "/openapi/bindDeviceLive",
                {
                    "token": self.token,
                    "deviceId": self.device_id,
                    "channelId": self.channel_id,
                    "streamId": stream_id
                }
            )
            
            if data:
                streams = data.get('streams', [])
                if streams:
                    stream = streams[0]
                    stream_url = stream.get('hls')
                    self.live_token = data.get('liveToken')
                    
                    # Update Cache with Quality Tag
                    ImouCamera._token_cache["stream_url"] = stream_url
                    ImouCamera._token_cache["live_token"] = self.live_token
                    # Store which stream ID (quality) this URL belongs to
                    ImouCamera._token_cache["stream_id"] = stream_id
                    
                    logger.info(f"✓ Device bound (streamId={stream_id}), stream URL: {stream_url[:60]}...")
                    return True
                else:
                    logger.warning("No streams in bind response")
                    return True # Return True to allow getLiveStreamInfo to try
            else:
                logger.warning("Bind failed (may already be bound)")
                return True
                
        except Exception as e:
            logger.error(f"Bind error: {e}", exc_info=True)
            return False
    
    def get_live_stream_url(self, quality: str = "SD") -> Optional[str]:
        """
        Получить URL live-трансляции
        quality: "HD" (main stream) или "SD" (sub stream)
        """
        try:
            stream_id = 0 if quality == "HD" else 1

            # ✅ CACHE CHECK: If cached URL exists and matches requested logic
            cached_url = ImouCamera._token_cache.get("stream_url")
            cached_sid = ImouCamera._token_cache.get("stream_id")
            
            if cached_url and cached_sid == stream_id:
                # Check token validity
                if self.get_access_token(): # This updates self.token if needed
                     logger.info(f"Using cached {quality} stream URL")
                     return cached_url

            if not self.get_access_token():
                return None
            
            if not self.bind_device_live(stream_id):
                logger.error("Failed to bind device")
                # Don't return None yet, try getting info anyway
            
            logger.info("Getting live stream info...")
            
            data = self._make_request(
                "/openapi/getLiveStreamInfo",
                {
                    "token": self.token,
                    "deviceId": self.device_id,
                    "channelId": self.channel_id
                }
            )
            
            if not data or 'streams' not in data:
                logger.error("No stream data received")
                if self.stream_url:
                    return self.stream_url
                return None
            
            streams = data['streams']
            
            for stream in streams:
                if stream.get('streamId') == stream_id and 'https' in stream.get('hls', '').lower():
                    self.stream_url = stream['hls']
                    logger.info(f"✓ HTTPS stream URL: {quality}")
                    return self.stream_url
            
            for stream in streams:
                if stream.get('streamId') == stream_id:
                    self.stream_url = stream['hls']
                    logger.info(f"✓ HTTP stream URL: {quality}")
                    return self.stream_url
            
            logger.error(f"No stream found for quality: {quality}")
            return None
            
        except Exception as e:
            logger.error(f"Stream URL error: {e}", exc_info=True)
            return None
    
    def open_stream(self, quality: str = "SD") -> Optional[cv2.VideoCapture]:
        """
        Открыть видеопоток с камеры
        quality: "HD" или "SD"
        """
        try:
            stream_url = self.get_live_stream_url(quality)
            
            if not stream_url:
                logger.error("Failed to get stream URL")
                return None
            
            logger.info("Opening video stream with OpenCV...")
            
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                logger.error("Failed to open video stream")
                return None
            
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.info(f"✓ Stream opened successfully! Frame: {frame.shape}")
                return cap
            else:
                logger.error("Cannot read frames from stream")
                cap.release()
                return None
            
        except Exception as e:
            logger.error(f"Stream open error: {e}", exc_info=True)
            return None

    def get_video_capture(self, quality: str = "SD"):
        """Совместимость: получить VideoCapture (аналог open_stream)"""
        return self.open_stream(quality)

    def test_connection(self) -> bool:
        """Полное тестирование подключения"""
        try:
            logger.info("="*60)
            logger.info("IMOU Camera Connection Test")
            logger.info("="*60)
            
            logger.info(f"Region: {self.region}")
            logger.info(f"Device ID: {self.device_id}")
            logger.info(f"Channel ID: {self.channel_id}")
            
            logger.info("\n[1/4] Testing access token...")
            if not self.get_access_token():
                logger.error("✗ Token test failed")
                return False
            logger.info("✓ Token obtained")
            
            logger.info("\n[2/4] Testing device binding...")
            if not self.bind_device_live(stream_id=1):
                logger.error("✗ Device binding failed")
                return False
            logger.info("✓ Device bound")
            
            logger.info("\n[3/4] Testing stream URL...")
            stream_url = self.get_live_stream_url("SD")
            if not stream_url:
                logger.error("✗ Stream URL test failed")
                return False
            logger.info(f"✓ Stream URL: {stream_url[:60]}...")
            
            logger.info("\n[4/4] Testing stream open...")
            cap = self.open_stream("SD")
            if cap:
                for i in range(3):
                    ret, frame = cap.read()
                    if ret:
                        logger.info(f"✓ Frame {i+1}: {frame.shape}")
                    else:
                        logger.warning(f"✗ Frame {i+1}: failed")
                cap.release()
                logger.info("\n✓ ALL TESTS PASSED!")
                logger.info("="*60)
                return True
            else:
                logger.error("✗ Stream open failed")
                return False
                
        except Exception as e:
            logger.error(f"Test error: {e}", exc_info=True)
            return False


__all__ = ["ImouCamera"]

def test_imou_camera():
    """Функция для тестирования"""
    print("\n" + "="*70)
    print("IMOU CAMERA CONNECTION TEST")
    print("="*70)
    print("\nIMPORTANT:")
    print("1. Make sure your system clock is synchronized")
    print("2. Device must be online in Imou Life app")
    print("3. Verify AppId and AppSecret")
    print()
    
    camera = ImouCamera()
    
    if camera.test_connection():
        print("\n" + "="*70)
        print("✓ SUCCESS! Camera is working correctly!")
        print("="*70)
        print("\nYou can now run: python run_imou.py")
    else:
        print("\n" + "="*70)
        print("✗ FAILED! Please check:")
        print("="*70)
        print("1. IMOU_APP_ID and IMOU_APP_SECRET are correct")
        print("2. IMOU_DEVICE_ID is correct")
        print("3. Device is online")
        print("4. System time is synchronized")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_imou_camera()