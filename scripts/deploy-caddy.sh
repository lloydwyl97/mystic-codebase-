#!/bin/bash

# Mystic Trading Platform - Caddy Deployment Script
set -e

echo "🚀 Mystic Trading Platform - Caddy Deployment"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Caddy is installed
if ! command -v caddy &> /dev/null; then
    print_error "Caddy is not installed. Please run: ./install-caddy.sh"
    exit 1
fi

print_status "Caddy is installed: $(caddy version)"

# Check if ports are available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        print_error "Port $1 is already in use. Please free up port $1."
        exit 1
    fi
}

echo "🔍 Checking port availability..."
check_port 80
check_port 443
check_port 8000

# Create necessary directories
print_status "Creating directories..."
sudo mkdir -p /var/www/mystic
sudo mkdir -p /var/log/caddy
sudo mkdir -p /etc/caddy

# Set proper permissions
sudo chown -R $USER:$USER /var/www/mystic
sudo chown -R $USER:$USER /var/log/caddy

# Build frontend
print_status "Building frontend..."
cd frontend
npm install
npm run build

# Copy frontend files
print_status "Copying frontend files..."
sudo cp -r dist/* /var/www/mystic/

# Copy Caddyfile
print_status "Configuring Caddy..."
sudo cp ../Caddyfile /etc/caddy/Caddyfile

# Create systemd service for Caddy
print_status "Creating Caddy service..."
sudo tee /etc/systemd/system/caddy.service > /dev/null <<EOF
[Unit]
Description=Caddy web server
Documentation=https://caddyserver.com/docs/
After=network.target network-online.target
Requires=network-online.target

[Service]
Type=notify
User=caddy
Group=caddy
ExecStart=/usr/bin/caddy run --environ --config /etc/caddy/Caddyfile
ExecReload=/usr/bin/caddy reload --config /etc/caddy/Caddyfile
TimeoutStopSec=5s
LimitNOFILE=1048576
LimitNPROC=512
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/etc/caddy /var/log/caddy /var/lib/caddy
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
EOF

# Create caddy user if it doesn't exist
if ! id "caddy" &>/dev/null; then
    print_status "Creating caddy user..."
    sudo useradd --system --shell /bin/false --home-dir /var/lib/caddy --group caddy
fi

# Set proper ownership
sudo chown -R caddy:caddy /var/log/caddy
sudo chown -R caddy:caddy /etc/caddy

# Enable and start Caddy
print_status "Starting Caddy service..."
sudo systemctl daemon-reload
sudo systemctl enable caddy
sudo systemctl start caddy

# Wait for Caddy to start
sleep 3

# Check if Caddy is running
if sudo systemctl is-active --quiet caddy; then
    print_status "Caddy is running successfully!"
else
    print_error "Caddy failed to start. Check logs with: sudo journalctl -u caddy"
    exit 1
fi

# Create backend startup script
print_status "Creating backend startup script..."
cat > start-backend.sh << 'EOF'
#!/bin/bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
gunicorn --config gunicorn.conf.py main:app
EOF

chmod +x start-backend.sh

# Create backend service
print_status "Creating backend service..."
sudo tee /etc/systemd/system/mystic-backend.service > /dev/null <<EOF
[Unit]
Description=Mystic Trading Backend
After=network.target

[Service]
Type=exec
User=$USER
WorkingDirectory=$(pwd)/backend
Environment=PATH=$(pwd)/backend/venv/bin
ExecStart=$(pwd)/backend/venv/bin/gunicorn --config gunicorn.conf.py main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Setup backend environment
print_status "Setting up backend environment..."
cd ../backend
if [ ! -f ".env" ]; then
    cp env.example .env
    print_warning "Please edit backend/.env with your production settings!"
fi

# Create Python virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

# Enable and start backend
print_status "Starting backend service..."
sudo systemctl daemon-reload
sudo systemctl enable mystic-backend
sudo systemctl start mystic-backend

# Wait for backend to start
sleep 5

# Health checks
echo "🏥 Running health checks..."

# Check Caddy
if curl -f http://localhost/ > /dev/null 2>&1; then
    print_status "Frontend is accessible at http://localhost"
else
    print_error "Frontend health check failed"
    sudo journalctl -u caddy --no-pager -n 20
fi

# Check backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_status "Backend is accessible at http://localhost:8000"
else
    print_error "Backend health check failed"
    sudo journalctl -u mystic-backend --no-pager -n 20
fi

# Check API proxy
if curl -f http://localhost/api/health > /dev/null 2>&1; then
    print_status "API proxy is working correctly"
else
    print_warning "API proxy check failed - backend might still be starting"
fi

echo ""
echo "🎉 Caddy deployment completed successfully!"
echo "=========================================="
echo "🌐 Frontend: http://localhost"
echo "🔧 Backend API: http://localhost:8000"
echo "🏥 Health Check: http://localhost/health"
echo "📊 API Docs: http://localhost:8000/docs"
echo ""
echo "📋 Useful commands:"
echo "  View Caddy logs: sudo journalctl -u caddy -f"
echo "  View backend logs: sudo journalctl -u mystic-backend -f"
echo "  Restart Caddy: sudo systemctl restart caddy"
echo "  Restart backend: sudo systemctl restart mystic-backend"
echo "  Stop all services: sudo systemctl stop caddy mystic-backend"
echo ""
echo "⚠️  Next steps:"
echo "  - Edit backend/.env with your production settings"
echo "  - Configure your domain in Caddyfile for production"
echo "  - Set up SSL certificates (Caddy handles this automatically)"
echo "  - Configure proper backup strategies"
