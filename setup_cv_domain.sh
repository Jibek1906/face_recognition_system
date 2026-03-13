#!/bin/bash
set -e

echo "Creating Nginx configuration for cv.pediatr.kg..."

cat << 'EOF' > /tmp/face_system_nginx.conf
server {
    listen 80;
    server_name cv.pediatr.kg;

    location / {
        proxy_pass https://127.0.0.1:9200;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Move the file to sites-available
sudo mv /tmp/face_system_nginx.conf /etc/nginx/sites-available/face_system

# Create a symlink in sites-enabled
sudo ln -sf /etc/nginx/sites-available/face_system /etc/nginx/sites-enabled/

echo "Testing Nginx configuration..."
sudo nginx -t

echo "Reloading Nginx..."
sudo systemctl reload nginx

echo "Obtaining SSL certificate using Certbot..."
sudo certbot --nginx -d cv.pediatr.kg --non-interactive --agree-tos --register-unsafely-without-email --redirect

echo "Done! The domain cv.pediatr.kg should now be securely bound to your face recognition system."
