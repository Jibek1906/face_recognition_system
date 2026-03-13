/**
 * visual_debug.js
 * Handles real-time face tracking visualization by drawing bounding boxes on a canvas overlay.
 */

class VisualDebugger {
    constructor(canvasId, videoId) {
        this.canvas = document.getElementById(canvasId);
        this.video = document.getElementById(videoId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.ws = null;
        this.tracks = {}; // id -> track data

        if (!this.canvas || !this.video) {
            console.error("VisualDebugger: Canvas or Video element not found!");
            return;
        }

        this.init();
    }

    init() {
        console.log("VisualDebugger: Initializing...");

        // Handle resizing
        window.addEventListener('resize', () => this.resizeCanvas());
        this.video.addEventListener('loadedmetadata', () => this.resizeCanvas());
        this.video.addEventListener('play', () => this.resizeCanvas());

        // Initial resize
        setTimeout(() => this.resizeCanvas(), 500);

        // Connect WebSocket
        this.connectWebSocket();

        // Start animation loop
        requestAnimationFrame(() => this.drawLoop());
    }

    resizeCanvas() {
        if (!this.video || !this.canvas) return;

        // Support both Video and Image elements
        let nativeW = this.video.videoWidth || this.video.naturalWidth;
        let nativeH = this.video.videoHeight || this.video.naturalHeight;

        // Fallback: If video is not ready or has 0 dimensions (common on mobile start)
        if (!nativeW || !nativeH) {
            nativeW = this.video.clientWidth || 1280;
            nativeH = this.video.clientHeight || 720;
        }

        const clientW = this.video.clientWidth;
        const clientH = this.video.clientHeight;

        // ✅ FIX: Use NATIVE resolution for canvas coordinate space
        if (this.canvas.width !== nativeW || this.canvas.height !== nativeH) {
            this.canvas.width = nativeW;
            this.canvas.height = nativeH;
        }

        // ✅ FIX: Match CSS size to video display size
        if (this.canvas.style.width !== clientW + 'px' || this.canvas.style.height !== clientH + 'px') {
            this.canvas.style.width = clientW + 'px';
            this.canvas.style.height = clientH + 'px';
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/tracks`;

        console.log(`VisualDebugger: Connecting to ${wsUrl}`);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log("VisualDebugger: WebSocket Connected");
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'tracks_update') {
                    this.updateTracks(data);
                    // ✅ INSTANT UI SYNC: Update sidebar list from WS data
                    if (window.updateActiveCameraVisitors && data.tracks) {
                        window.updateActiveCameraVisitors(data.tracks);
                    }
                }
            } catch (e) {
                console.error("VisualDebugger: Parse error", e);
            }
        };

        this.ws.onclose = () => {
            console.log("VisualDebugger: WebSocket Disconnected. Reconnecting in 3s...");
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (err) => {
            console.error("VisualDebugger: WebSocket Error", err);
        };
    }

    updateTracks(data) {
        // Update valid tracks
        if (data.tracks) {
            data.tracks.forEach(track => {
                this.tracks[track.id] = {
                    ...track,
                    lastUpdate: Date.now()
                };
            });
        }

        // Remove stale tracks
        if (data.removed) {
            data.removed.forEach(id => {
                delete this.tracks[id];
            });
        }
    }

    drawLoop() {
        if (!this.ctx || !this.canvas) return;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const now = Date.now();
        const displayW = this.canvas.width;
        const displayH = this.canvas.height;

        const videoNativeW = this.video.videoWidth || this.video.naturalWidth || 1280; // Fallback
        const videoNativeH = this.video.videoHeight || this.video.naturalHeight || 720;

        // ✅ FIX: Scale is always 1.0 because canvas.width == nativeW
        // We handle display scaling via CSS
        const scaleX = 1.0;
        const scaleY = 1.0;

        // Interpolation Factor (Adjust for smoothness vs latency)
        // 0.2 = very smooth but laggy, 0.8 = responsive but jittery
        // INCREASED to 0.7 for responsiveness per user request
        const LERP_FACTOR = 0.7;

        // We will perform update AND draw in one loop, but we need to know collisions.
        // Let's settle for: Draw Known (Green/Blue) ON TOP of Unknown (Red).
        // AND don't draw Red if it overlaps Known.

        // 1. Separate lists
        const tracks = Object.values(this.tracks);
        const knowns = [];
        const unknowns = [];

        tracks.forEach(track => {
            // Prune old tracks locally if WS missed a removal (cleanup)
            // INCREASED buffer to prevent flickering: 2000ms -> 3000ms
            if (now - track.lastUpdate > 3000) {
                delete this.tracks[track.id];
                return; // Skip this track
            }

            let tx, ty, tw, th;
            // ... (Calculation Logic) ...
            if (track.bbox_norm) {
                // ✅ Use NORMALIZED coordinates with OBJECT-FIT: COVER math

                // 1. Calculate aspect ratios
                const videoAspect = videoNativeW / videoNativeH;
                const displayAspect = displayW / displayH;

                let renderW, renderH, offsetX, offsetY;

                if (displayAspect > videoAspect) {
                    // Container is WIDER than video -> Video is fit to WIDTH, Top/Bottom cropped
                    renderW = displayW;
                    renderH = displayW / videoAspect;
                    offsetX = 0;
                    offsetY = (displayH - renderH) / 2;
                } else {
                    // Container is TALLER than video -> Video is fit to HEIGHT, Left/Right cropped
                    renderW = displayH * videoAspect;
                    renderH = displayH;
                    offsetX = (displayW - renderW) / 2;
                    offsetY = 0;
                }

                // 2. Map normalized coords to RENDERED dimensions
                tx = (track.bbox_norm[0] * renderW) + offsetX;
                ty = (track.bbox_norm[1] * renderH) + offsetY;
                tw = (track.bbox_norm[2] * renderW);
                th = (track.bbox_norm[3] * renderH);
            } else {
                // Fallback to legacy pixel scaling (should not happen with new backend)
                let [x, y, w, h] = track.bbox;
                tx = x * scaleX;
                ty = y * scaleY;
                tw = w * scaleX;
                th = h * scaleY;
            }

            // Init smoothed coords if missing
            if (!track.smoothX) {
                track.smoothX = tx;
                track.smoothY = ty;
                track.smoothW = tw;
                track.smoothH = th;
            }

            // Apply Linear Interpolation (LERP)
            track.smoothX += (tx - track.smoothX) * LERP_FACTOR;
            track.smoothY += (ty - track.smoothY) * LERP_FACTOR;
            track.smoothW += (tw - track.smoothW) * LERP_FACTOR;
            track.smoothH += (th - track.smoothH) * LERP_FACTOR;

            track.drawBox = { x: track.smoothX, y: track.smoothY, w: track.smoothW, h: track.smoothH };

            if (track.is_employee || track.recognized || track.person_id) {
                knowns.push(track);
            } else {
                unknowns.push(track);
            }
        });

        // Helper IoU
        const getIoU = (b1, b2) => {
            let x1 = Math.max(b1.x, b2.x), y1 = Math.max(b1.y, b2.y);
            let x2 = Math.min(b1.x + b1.w, b2.x + b2.w), y2 = Math.min(b1.y + b1.h, b2.y + b2.h);
            if (x2 < x1 || y2 < y1) return 0;
            let inter = (x2 - x1) * (y2 - y1);
            return inter / (b1.w * b1.h + b2.w * b2.h - inter + 1e-6);
        };

        // Draw Loop
        // 1. Draw Unknowns ONLY if they don't overlap Knowns
        unknowns.forEach(u => {
            // SMOOTHING: If track is "stale" (no update for >500ms), show it faded
            const age = now - u.lastUpdate;
            // Remove effectively if stale for too long (visual cleanup)
            if (age > 500) { delete this.tracks[u.id]; return; }

            const isStale = age > 500;
            const opacity = isStale ? Math.max(0.3, 1 - (age - 500) / 1000) : 1;

            let isGhost = false;
            for (const k of knowns) {
                if (getIoU(u.drawBox, k.drawBox) > 0.4) {
                    isGhost = true; break;
                }
            }
            if (isGhost) return; // SKIP DRAWING GHOST

            this.ctx.globalAlpha = opacity;

            // Draw Label (Name / ID)
            // Fix: Flip text scale so it reads correctly on mirrored canvas
            this.ctx.save();
            const textX = u.drawBox.x + u.drawBox.w / 2;
            const textY = u.drawBox.y - 10;

            this.ctx.translate(textX, textY);
            this.ctx.scale(-1, 1); // Flip horizonal

            // Draw Background
            const text = u.name || `ID ${u.id}`;
            this.ctx.font = "bold 14px Manrope, sans-serif";
            const metrics = this.ctx.measureText(text);
            const bgW = metrics.width + 16;
            const bgH = 24;

            this.ctx.fillStyle = '#ef4444'; // Red for unknowns

            // Safe Round Rect (Manual)
            this.ctx.beginPath();
            const x = -bgW / 2, y = -bgH + 5, w = bgW, h = bgH, r = 6;
            this.ctx.moveTo(x + r, y);
            this.ctx.arcTo(x + w, y, x + w, y + h, r);
            this.ctx.arcTo(x + w, y + h, x, y + h, r);
            this.ctx.arcTo(x, y + h, x, y, r);
            this.ctx.arcTo(x, y, x + w, y, r);
            this.ctx.closePath();
            this.ctx.fill();

            // Draw Text
            this.ctx.fillStyle = "#ffffff";
            this.ctx.textAlign = "center";
            this.ctx.fillText(text, 0, 0);

            this.ctx.restore();

            // Draw Red Box
            this.ctx.strokeStyle = '#ef4444';
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.strokeRect(u.drawBox.x, u.drawBox.y, u.drawBox.w, u.drawBox.h);
        });

        // 2. Draw Knowns (Always)
        knowns.forEach(k => {
            // SMOOTHING: If track is "stale" (no update for >500ms), show it faded
            const age = now - k.lastUpdate;
            // Remove effectively if stale for too long (visual cleanup)
            if (age > 500) { delete this.tracks[k.id]; return; }

            const isStale = age > 500;
            const opacity = isStale ? Math.max(0.3, 1 - (age - 500) / 1000) : 1;

            let color = '#10b981'; // Green
            if (k.is_employee) color = '#fbbf24'; // Gold
            else if (k.recognized || (k.similarity && k.similarity > 0.4)) color = '#10b981'; // Green

            if (k.person_id) color = '#10b981'; // Green (Confirmed)

            this.ctx.globalAlpha = opacity;

            // Draw Label (Name / ID)
            // Fix: Flip text scale so it reads correctly on mirrored canvas
            this.ctx.save();
            const textX = k.drawBox.x + k.drawBox.w / 2;
            const textY = k.drawBox.y - 10;

            this.ctx.translate(textX, textY);
            this.ctx.scale(-1, 1); // Flip horizonal

            // Draw Background
            const text = k.name || (k.person_id ? `Client ${k.person_id}` : `ID ${k.id}`);
            this.ctx.font = "bold 14px Manrope, sans-serif";
            const metrics = this.ctx.measureText(text);
            const bgW = metrics.width + 16;
            const bgH = 24;

            this.ctx.fillStyle = color;

            // Safe Round Rect (Manual)
            this.ctx.beginPath();
            const x = -bgW / 2, y = -bgH + 5, w = bgW, h = bgH, r = 6;
            this.ctx.moveTo(x + r, y);
            this.ctx.arcTo(x + w, y, x + w, y + h, r);
            this.ctx.arcTo(x + w, y + h, x, y + h, r);
            this.ctx.arcTo(x, y + h, x, y, r);
            this.ctx.arcTo(x, y, x + w, y, r);
            this.ctx.closePath();
            this.ctx.fill();

            // Draw Text
            this.ctx.fillStyle = "#ffffff";
            if (color === '#fbbf24') this.ctx.fillStyle = "#000000";

            this.ctx.textAlign = "center";
            this.ctx.fillText(text, 0, 0);

            this.ctx.restore();

            // Draw Box
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.strokeRect(k.drawBox.x, k.drawBox.y, k.drawBox.w, k.drawBox.h);
        });

        requestAnimationFrame(() => this.drawLoop());
    }
}

// Global initialization
window.visualDebugger = new VisualDebugger('camera-overlay', 'live-video');
