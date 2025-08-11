# ===== NLP AGENT SYSTEM LAUNCHER =====
# Launches the complete NLP agent system for sentiment analysis

Write-Host "🧠 Launching NLP Agent System..." -ForegroundColor Cyan

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker info | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check if required files exist
function Test-RequiredFiles {
    $requiredFiles = @(
        "docker-compose.yml",
        "backend/agents/news_sentiment_agent.py",
        "backend/agents/social_media_agent.py", 
        "backend/agents/market_sentiment_agent.py",
        "backend/agents/nlp_orchestrator.py"
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Write-Host "❌ Required file not found: $file" -ForegroundColor Red
            return $false
        }
    }
    return $true
}

# Function to start Redis if not running
function Start-RedisIfNeeded {
    Write-Host "🔍 Checking Redis status..." -ForegroundColor Yellow
    
    try {
        $redisStatus = docker ps --filter "name=redis" --format "table {{.Names}}\t{{.Status}}"
        if ($redisStatus -like "*redis*") {
            Write-Host "✅ Redis is already running" -ForegroundColor Green
        } else {
            Write-Host "🚀 Starting Redis..." -ForegroundColor Yellow
            docker-compose up -d redis
            Start-Sleep -Seconds 5
        }
    }
    catch {
        Write-Host "❌ Error checking Redis status: $_" -ForegroundColor Red
        throw
    }
}

# Function to start NLP services
function Start-NLPServices {
    Write-Host "🚀 Starting NLP Agent Services..." -ForegroundColor Yellow
    
    try {
        # Start core NLP services
        $nlpServices = @(
            "news-sentiment-agent",
            "social-media-agent", 
            "market-sentiment-agent",
            "nlp-orchestrator"
        )
        
        foreach ($service in $nlpServices) {
            Write-Host "📦 Starting $service..." -ForegroundColor Yellow
            docker-compose up -d $service
            Start-Sleep -Seconds 3
        }
        
        Write-Host "✅ All NLP services started successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Error starting NLP services: $_" -ForegroundColor Red
        throw
    }
}

# Function to check service health
function Test-ServiceHealth {
    Write-Host "🏥 Checking service health..." -ForegroundColor Yellow
    
    $services = @(
        @{Name="News Sentiment Agent"; Container="mystic-news-sentiment-agent"},
        @{Name="Social Media Agent"; Container="mystic-social-media-agent"},
        @{Name="Market Sentiment Agent"; Container="mystic-market-sentiment-agent"},
        @{Name="NLP Orchestrator"; Container="mystic-nlp-orchestrator"}
    )
    
    $healthyServices = 0
    
    foreach ($service in $services) {
        try {
            $status = docker inspect --format='{{.State.Status}}' $service.Container 2>$null
            if ($status -eq "running") {
                Write-Host "✅ $($service.Name): Running" -ForegroundColor Green
                $healthyServices++
            } else {
                Write-Host "❌ $($service.Name): $status" -ForegroundColor Red
            }
        }
        catch {
            Write-Host "❌ $($service.Name): Not found" -ForegroundColor Red
        }
    }
    
    return $healthyServices -eq $services.Count
}

# Function to display service logs
function Show-ServiceLogs {
    Write-Host "📋 Recent service logs:" -ForegroundColor Cyan
    
    $services = @(
        "mystic-news-sentiment-agent",
        "mystic-social-media-agent",
        "mystic-market-sentiment-agent", 
        "mystic-nlp-orchestrator"
    )
    
    foreach ($service in $services) {
        Write-Host "`n📝 $service logs:" -ForegroundColor Yellow
        try {
            docker logs --tail=5 $service 2>$null
        }
        catch {
            Write-Host "No logs available for $service" -ForegroundColor Gray
        }
    }
}

# Function to display system status
function Show-SystemStatus {
    Write-Host "`n📊 NLP System Status:" -ForegroundColor Cyan
    
    # Show running containers
    Write-Host "`n🐳 Running Containers:" -ForegroundColor Yellow
    docker ps --filter "name=mystic-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    # Show Redis status
    Write-Host "`n🔴 Redis Status:" -ForegroundColor Yellow
    try {
        $redisInfo = docker exec mystic-redis redis-cli info server 2>$null
        if ($redisInfo) {
            Write-Host "✅ Redis is operational" -ForegroundColor Green
        } else {
            Write-Host "❌ Redis is not responding" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "❌ Cannot connect to Redis" -ForegroundColor Red
    }
}

# Function to provide usage instructions
function Show-UsageInstructions {
    Write-Host "`n📖 NLP System Usage Instructions:" -ForegroundColor Cyan
    
    Write-Host @"
🧠 NLP Agent System is now running!

Available Services:
• News Sentiment Agent: Analyzes financial news sentiment
• Social Media Agent: Monitors social media sentiment  
• Market Sentiment Agent: Aggregates sentiment from all sources
• NLP Orchestrator: Coordinates all NLP agents

Key Features:
• Real-time sentiment analysis from news sources
• Social media monitoring (Twitter, Reddit, Telegram)
• Market sentiment aggregation and Fear & Greed Index
• Unified sentiment signals for trading decisions

Monitoring Commands:
• View logs: docker logs mystic-nlp-orchestrator
• Check status: docker ps --filter name=mystic-
• Stop system: docker-compose down

Integration:
• NLP agents communicate with existing trading agents
• Sentiment data feeds into strategy and risk decisions
• Real-time sentiment updates via Redis channels

For more information, see: docs/NLP_AGENT_SYSTEM_README.md
"@ -ForegroundColor White
}

# Main execution
try {
    Write-Host "🧠 NLP Agent System Launcher" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    
    # Check Docker
    if (-not (Test-DockerRunning)) {
        Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
        exit 1
    }
    
    # Check required files
    if (-not (Test-RequiredFiles)) {
        Write-Host "❌ Required files are missing. Please ensure all NLP agent files are present." -ForegroundColor Red
        exit 1
    }
    
    # Start Redis
    Start-RedisIfNeeded
    
    # Start NLP services
    Start-NLPServices
    
    # Wait for services to initialize
    Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    
    # Check health
    if (Test-ServiceHealth) {
        Write-Host "`n🎉 NLP Agent System launched successfully!" -ForegroundColor Green
        
        # Show system status
        Show-SystemStatus
        
        # Show usage instructions
        Show-UsageInstructions
        
        # Show recent logs
        Show-ServiceLogs
        
    } else {
        Write-Host "`n⚠️ Some services may not be healthy. Check logs for details." -ForegroundColor Yellow
        Show-ServiceLogs
    }
    
}
catch {
    Write-Host "`n❌ Error launching NLP Agent System: $_" -ForegroundColor Red
    Write-Host "Please check the error details above and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host "`n✅ NLP Agent System launcher completed!" -ForegroundColor Green 