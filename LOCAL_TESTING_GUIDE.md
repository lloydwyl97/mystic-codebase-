# 🖥️ LOCAL TESTING GUIDE

## 📋 **AVAILABLE SCRIPTS & BATCH FILES**

### **🚀 Main Startup Scripts**

#### **Root Directory Scripts:**
- `quick_test.ps1` - Quick system test
- `start_backend.ps1` - Start backend service
- `run_all_entrypoints.py` - Run all entry points
- `install_everything.ps1` - Complete installation
- `run_dashboard_docker.bat` - Dashboard Docker startup
- `run_dashboard_docker.ps1` - Dashboard Docker PowerShell

#### **Scripts Directory (`scripts/`):**

**🔧 Setup Scripts:**
- `setup-dev.ps1` - Development environment setup
- `setup-python310.ps1` - Python 3.10 setup
- `setup-auto-withdraw.ps1` - Auto-withdraw setup
- `install-requirements.ps1` - Install Python requirements
- `install-missing-packages.ps1` - Install missing packages

**🚀 Launch Scripts:**
- `launch.ps1` - Main platform launch
- `launch-live.ps1` - Live trading launch
- `launch_advanced_ai_system.ps1` - Advanced AI system
- `launch_quantum_system.ps1` - Quantum computing system
- `launch_phase5_agents.ps1` - Phase 5 monitoring agents
- `launch_computer_vision_system.ps1` - Computer vision
- `launch_nlp_system.ps1` - Natural language processing

**⚡ Service Scripts:**
- `start-backend.bat` - Backend service (Windows)
- `start-frontend.bat` - Frontend service (Windows)
- `start-redis.bat` - Redis server (Windows)
- `start-live-trading.ps1` - Live trading system
- `start-modular-live.ps1` - Modular live system
- `start-docker-modular.ps1` - Docker modular system

**🐳 Docker Scripts:**
- `docker-build-clean.ps1` - Docker build and cleanup
- `docker-backend.ps1` - Backend Docker container
- `docker-frontend.ps1` - Frontend Docker container
- `docker-compose-full.ps1` - Full Docker Compose
- `start-advanced-ai-docker.ps1` - AI Docker system

**🔧 Utility Scripts:**
- `fix-logic-errors.ps1` - Fix logic errors
- `rebuild-agents-fix.ps1` - Rebuild agents
- `test-live-connections.ps1` - Test connections
- `run-ai-strategies.ps1` - Run AI strategies
- `run-auto-withdraw.ps1` - Auto-withdraw system
- `create-desktop-shortcut.ps1` - Create shortcuts
- `cleanup-docker.ps1` - Docker cleanup
- `build-base-image.ps1` - Build base image
- `build-optimized.ps1` - Optimized build

**🌐 Web Server Scripts:**
- `install-caddy.sh` - Install Caddy web server
- `deploy-caddy.sh` - Deploy Caddy configuration
- `Caddyfile` - Caddy configuration
- `Caddyfile.production` - Production Caddy config

## 🔥 **FIREWALL PORT REQUIREMENTS**

### **🎯 CRITICAL PORTS (Must Allow)**

#### **Core Trading System:**
```
9000 - Main Backend (FastAPI)
8501 - Super Dashboard (Streamlit)
8000 - Dashboard API
8080 - Trade Logging/Autobuy Dashboard
```

#### **AI & Analytics:**
```
8001 - AI Service
8002 - AI Processor
8003 - Visualization
8004 - AI Trade Engine
```

#### **Cache & Database:**
```
6379 - Redis (Primary)
6380 - Redis Cluster 1 (if using cluster)
6381 - Redis Cluster 2 (if using cluster)
6382 - Redis Cluster 3 (if using cluster)
```

#### **Message Queue:**
```
5672 - RabbitMQ
15672 - RabbitMQ Management (Web UI)
```

### **🔬 OPTIONAL PORTS (Advanced Features)**

#### **Quantum Computing:**
```
8087 - Quantum Trading Engine
8088 - Quantum Optimization
8089 - Quantum Machine Learning
8106 - QKD Alice
```

#### **5G Network Simulation:**
```
8093 - 5G Core
8094 - 5G RAN
8095 - 5G Slice Manager
```

#### **Blockchain Mining:**
```
8099 - Bitcoin Miner
8100 - Ethereum Miner
8101 - Mining Pool
```

#### **Satellite Operations:**
```
8096 - Satellite Receiver
8097 - Satellite Processor
8098 - Satellite Analytics
```

#### **Edge Computing:**
```
8090 - Edge Node 1
8091 - Edge Node 2
8092 - Edge Orchestrator
```

#### **Advanced AI:**
```
8102 - AI Super Master
```

#### **Monitoring Stack:**
```
9090 - Prometheus
3000 - Grafana
```

#### **Infrastructure:**
```
2181 - Zookeeper
8265 - Ray Dashboard
10001 - Ray Port
```

## 🛡️ **WINDOWS FIREWALL CONFIGURATION**

### **PowerShell Commands to Allow Ports:**

```powershell
# Core Trading Ports
New-NetFirewallRule -DisplayName "Mystic Backend" -Direction Inbound -Protocol TCP -LocalPort 9000 -Action Allow
New-NetFirewallRule -DisplayName "Mystic Dashboard" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow
New-NetFirewallRule -DisplayName "Mystic API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
New-NetFirewallRule -DisplayName "Mystic Trade Logging" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow

# AI Services
New-NetFirewallRule -DisplayName "Mystic AI Service" -Direction Inbound -Protocol TCP -LocalPort 8001 -Action Allow
New-NetFirewallRule -DisplayName "Mystic AI Processor" -Direction Inbound -Protocol TCP -LocalPort 8002 -Action Allow
New-NetFirewallRule -DisplayName "Mystic Visualization" -Direction Inbound -Protocol TCP -LocalPort 8003 -Action Allow
New-NetFirewallRule -DisplayName "Mystic AI Trade Engine" -Direction Inbound -Protocol TCP -LocalPort 8004 -Action Allow

# Redis
New-NetFirewallRule -DisplayName "Mystic Redis" -Direction Inbound -Protocol TCP -LocalPort 6379 -Action Allow
New-NetFirewallRule -DisplayName "Mystic Redis Cluster 1" -Direction Inbound -Protocol TCP -LocalPort 6380 -Action Allow
New-NetFirewallRule -DisplayName "Mystic Redis Cluster 2" -Direction Inbound -Protocol TCP -LocalPort 6381 -Action Allow
New-NetFirewallRule -DisplayName "Mystic Redis Cluster 3" -Direction Inbound -Protocol TCP -LocalPort 6382 -Action Allow

# Message Queue
New-NetFirewallRule -DisplayName "Mystic RabbitMQ" -Direction Inbound -Protocol TCP -LocalPort 5672 -Action Allow
New-NetFirewallRule -DisplayName "Mystic RabbitMQ Management" -Direction Inbound -Protocol TCP -LocalPort 15672 -Action Allow

# Monitoring
New-NetFirewallRule -DisplayName "Mystic Prometheus" -Direction Inbound -Protocol TCP -LocalPort 9090 -Action Allow
New-NetFirewallRule -DisplayName "Mystic Grafana" -Direction Inbound -Protocol TCP -LocalPort 3000 -Action Allow
```

### **Batch Script for Easy Firewall Setup:**

```batch
@echo off
echo Setting up Mystic Trading Platform Firewall Rules...

REM Core Trading Ports
netsh advfirewall firewall add rule name="Mystic Backend" dir=in action=allow protocol=TCP localport=9000
netsh advfirewall firewall add rule name="Mystic Dashboard" dir=in action=allow protocol=TCP localport=8501
netsh advfirewall firewall add rule name="Mystic API" dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="Mystic Trade Logging" dir=in action=allow protocol=TCP localport=8080

REM AI Services
netsh advfirewall firewall add rule name="Mystic AI Service" dir=in action=allow protocol=TCP localport=8001
netsh advfirewall firewall add rule name="Mystic AI Processor" dir=in action=allow protocol=TCP localport=8002
netsh advfirewall firewall add rule name="Mystic Visualization" dir=in action=allow protocol=TCP localport=8003
netsh advfirewall firewall add rule name="Mystic AI Trade Engine" dir=in action=allow protocol=TCP localport=8004

REM Redis
netsh advfirewall firewall add rule name="Mystic Redis" dir=in action=allow protocol=TCP localport=6379
netsh advfirewall firewall add rule name="Mystic Redis Cluster 1" dir=in action=allow protocol=TCP localport=6380
netsh advfirewall firewall add rule name="Mystic Redis Cluster 2" dir=in action=allow protocol=TCP localport=6381
netsh advfirewall firewall add rule name="Mystic Redis Cluster 3" dir=in action=allow protocol=TCP localport=6382

REM Message Queue
netsh advfirewall firewall add rule name="Mystic RabbitMQ" dir=in action=allow protocol=TCP localport=5672
netsh advfirewall firewall add rule name="Mystic RabbitMQ Management" dir=in action=allow protocol=TCP localport=15672

REM Monitoring
netsh advfirewall firewall add rule name="Mystic Prometheus" dir=in action=allow protocol=TCP localport=9090
netsh advfirewall firewall add rule name="Mystic Grafana" dir=in action=allow protocol=TCP localport=3000

echo Firewall rules configured successfully!
pause
```

## 🚀 **LOCAL TESTING PROCEDURE**

### **Step 1: Environment Setup**
```powershell
# Run setup script
.\scripts\setup-dev.ps1

# Install requirements
.\scripts\install-requirements.ps1
```

### **Step 2: Firewall Configuration**
```powershell
# Run firewall setup (as Administrator)
.\scripts\setup-firewall.ps1
```

### **Step 3: Start Core Services**
```powershell
# Start Redis
.\scripts\start-redis.bat

# Start Backend
.\scripts\start-backend.bat

# Start Dashboard
.\scripts\start-frontend.bat
```

### **Step 4: Test Core Functionality**
```powershell
# Test backend
curl http://localhost:9000/health

# Test dashboard
Start-Process "http://localhost:8501"

# Test Redis
redis-cli ping
```

### **Step 5: Launch Advanced Features**
```powershell
# Launch AI system
.\scripts\launch_advanced_ai_system.ps1

# Launch quantum system
.\scripts\launch_quantum_system.ps1

# Launch live trading
.\scripts\start-live-trading.ps1
```

## 📊 **PORT USAGE SUMMARY**

### **Essential Ports (Always Needed):**
- **9000** - Main backend
- **8501** - Dashboard
- **6379** - Redis
- **8000** - API

### **AI & Analytics (Recommended):**
- **8001-8004** - AI services
- **5672** - Message queue
- **15672** - Queue management

### **Advanced Features (Optional):**
- **8087-8089** - Quantum computing
- **8093-8095** - 5G simulation
- **8099-8101** - Blockchain mining
- **9090, 3000** - Monitoring

## ⚠️ **IMPORTANT NOTES**

1. **Run PowerShell as Administrator** for firewall configuration
2. **Start Redis first** before other services
3. **Check port availability** before starting services
4. **Use localhost** for all connections (127.0.0.1)
5. **Monitor system resources** during testing

## 🔧 **TROUBLESHOOTING**

### **Common Issues:**
- **Port already in use**: Stop conflicting services
- **Firewall blocking**: Run firewall setup as Administrator
- **Redis connection failed**: Start Redis service first
- **Python environment**: Activate virtual environment

### **Testing Commands:**
```powershell
# Check if ports are open
netstat -an | findstr :9000
netstat -an | findstr :8501

# Test Redis connection
redis-cli -h localhost -p 6379 ping

# Check service status
Get-Process | Where-Object {$_.ProcessName -like "*python*"}
```

**All scripts are ready for local testing on your laptop!** 