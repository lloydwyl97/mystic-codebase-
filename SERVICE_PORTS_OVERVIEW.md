# 🚀 SERVICE PORTS OVERVIEW

## 📊 **CORE SERVICES**

### **Backend Services**
- **Main Backend** - `9000` (Main FastAPI application)
- **Dashboard API** - `8000` (Dashboard API service)
- **Trade Logging** - `8080` (Trade logging service)
- **Autobuy Dashboard** - `8080` (Autobuy dashboard)

### **AI Services**
- **AI Service** - `8001` (AI processing service)
- **AI Processor** - `8002` (AI strategy generator)
- **AI Trade Engine** - `8004` (AI trading engine service)
- **Visualization** - `8003` (Data visualization service)

### **Dashboard Services**
- **Super Dashboard** - `8501` (Streamlit dashboard)
- **Visualization Dashboard** - `8080` (Plotly dashboard)

## 🔬 **QUANTUM SERVICES**

### **Quantum Computing**
- **Quantum Trading Engine** - `8087` (Quantum trading engine)
- **Quantum Optimization** - `8088` (Quantum optimization)
- **Quantum Machine Learning** - `8089` (Quantum ML)

### **Quantum Key Distribution**
- **QKD Alice** - `8106` (Quantum key distribution)

## 🌐 **5G SERVICES**

### **5G Network**
- **5G Core** - `8093` (5G core network)
- **5G RAN** - `8094` (5G radio access network)
- **5G Slice Manager** - `8095` (5G network slicing)

## ⛓️ **BLOCKCHAIN SERVICES**

### **Mining Operations**
- **Bitcoin Miner** - `8099` (Bitcoin mining)
- **Ethereum Miner** - `8100` (Ethereum mining)
- **Mining Pool** - `8101` (Mining pool management)

## 🛰️ **SATELLITE SERVICES**

### **Satellite Operations**
- **Satellite Receiver** - `8096` (Satellite data receiver)
- **Satellite Processor** - `8097` (Satellite data processor)
- **Satellite Analytics** - `8098` (Satellite analytics)

## 🔄 **EDGE COMPUTING**

### **Edge Computing**
- **Edge Node 1** - `8090` (Edge computing node 1)
- **Edge Node 2** - `8091` (Edge computing node 2)
- **Edge Orchestrator** - `8092` (Edge computing orchestrator)

## 🤖 **AI SUPERCOMPUTER**

### **AI Supercomputer**
- **AI Super Master** - `8102` (AI supercomputer master)

## 📊 **MONITORING & METRICS**

### **Monitoring Services**
- **Prometheus** - `9090` (Metrics collection)
- **Grafana** - `3000` (Monitoring dashboard)

## 🔌 **MESSAGE QUEUE & CACHE**

### **Message Queue**
- **RabbitMQ** - `5672` (Message queue)
- **RabbitMQ Management** - `15672` (RabbitMQ web interface)

### **Cache & Storage**
- **Redis** - `6379` (Primary cache)
- **Redis Cluster 1** - `6380` (Redis cluster node 1)
- **Redis Cluster 2** - `6381` (Redis cluster node 2)
- **Redis Cluster 3** - `6382` (Redis cluster node 3)

## 🗄️ **INFRASTRUCTURE**

### **Infrastructure Services**
- **Zookeeper** - `2181` (Distributed coordination)
- **Ray Head** - `8265` (Ray dashboard)
- **Ray Port** - `10001` (Ray distributed computing)

## 📋 **PORT SUMMARY BY CATEGORY**

### **Core Trading (4 ports)**
```
9000 - Main Backend
8000 - Dashboard API
8080 - Trade Logging/Autobuy Dashboard
8501 - Super Dashboard
```

### **AI & Analytics (4 ports)**
```
8001 - AI Service
8002 - AI Processor
8003 - Visualization
8004 - AI Trade Engine
```

### **Quantum Computing (4 ports)**
```
8087 - Quantum Trading Engine
8088 - Quantum Optimization
8089 - Quantum Machine Learning
8106 - QKD Alice
```

### **5G Network (3 ports)**
```
8093 - 5G Core
8094 - 5G RAN
8095 - 5G Slice Manager
```

### **Blockchain Mining (3 ports)**
```
8099 - Bitcoin Miner
8100 - Ethereum Miner
8101 - Mining Pool
```

### **Satellite Operations (3 ports)**
```
8096 - Satellite Receiver
8097 - Satellite Processor
8098 - Satellite Analytics
```

### **Edge Computing (3 ports)**
```
8090 - Edge Node 1
8091 - Edge Node 2
8092 - Edge Orchestrator
```

### **Advanced AI (1 port)**
```
8102 - AI Super Master
```

### **Monitoring (2 ports)**
```
9090 - Prometheus
3000 - Grafana
```

### **Message Queue (2 ports)**
```
5672 - RabbitMQ
15672 - RabbitMQ Management
```

### **Cache & Storage (4 ports)**
```
6379 - Redis (Primary)
6380 - Redis Cluster 1
6381 - Redis Cluster 2
6382 - Redis Cluster 3
```

### **Infrastructure (3 ports)**
```
2181 - Zookeeper
8265 - Ray Dashboard
10001 - Ray Port
```

## 🎯 **TOTAL PORTS: 37**

### **Port Range Distribution:**
- **8000-8099**: 15 ports (Core services, AI, Quantum, 5G, Blockchain, Satellite, Edge)
- **8100-8199**: 3 ports (Advanced AI, Blockchain)
- **3000-3999**: 1 port (Grafana)
- **5000-5999**: 1 port (RabbitMQ)
- **6000-6999**: 4 ports (Redis cluster)
- **9000-9999**: 1 port (Main backend)
- **10000+**: 1 port (Ray)
- **2000-2999**: 1 port (Zookeeper)
- **15000+**: 1 port (RabbitMQ Management)

## 🔧 **DOCKER COMPOSE CONFIGURATION**

All services are configured in `docker-compose.yml` with:
- **Health checks** for critical services
- **Volume mounts** for data persistence
- **Network isolation** via `mystic-network`
- **Environment variables** for configuration
- **Dependencies** between services

## 🚀 **DEPLOYMENT NOTES**

### **Critical Services (Must Start First):**
1. **Redis** (6379) - Cache and messaging
2. **Backend** (9000) - Main application
3. **Dashboard** (8501) - User interface

### **Optional Services:**
- **Quantum services** (8087-8089) - Experimental
- **5G services** (8093-8095) - Network simulation
- **Blockchain services** (8099-8101) - Mining simulation
- **Satellite services** (8096-8098) - Data processing
- **Edge services** (8090-8092) - Distributed computing

### **Monitoring Stack:**
- **Prometheus** (9090) + **Grafana** (3000) - Metrics and monitoring
- **RabbitMQ** (5672) + **Management** (15672) - Message queue

**All services are containerized and ready for production deployment with comprehensive monitoring and health checks.** 