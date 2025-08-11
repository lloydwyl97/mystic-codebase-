# setup-firewall.ps1 – Firewall Configuration for Mystic Trading Platform
# Run as Administrator

param(
    [switch]$RemoveRules,
    [switch]$ListRules,
    [switch]$TestPorts
)

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Add-MysticFirewallRules {
    Write-ColorOutput "🔥 Setting up Mystic Trading Platform Firewall Rules..." Green
    
    # Core Trading Ports
    $corePorts = @(
        @{Port=9000; Name="Mystic Backend"},
        @{Port=8501; Name="Mystic Dashboard"},
        @{Port=8000; Name="Mystic API"},
        @{Port=8080; Name="Mystic Trade Logging"}
    )
    
    # AI & Analytics Ports
    $aiPorts = @(
        @{Port=8001; Name="Mystic AI Service"},
        @{Port=8002; Name="Mystic AI Processor"},
        @{Port=8003; Name="Mystic Visualization"},
        @{Port=8004; Name="Mystic AI Trade Engine"}
    )
    
    # Cache & Database Ports
    $cachePorts = @(
        @{Port=6379; Name="Mystic Redis"},
        @{Port=6380; Name="Mystic Redis Cluster 1"},
        @{Port=6381; Name="Mystic Redis Cluster 2"},
        @{Port=6382; Name="Mystic Redis Cluster 3"}
    )
    
    # Message Queue Ports
    $queuePorts = @(
        @{Port=5672; Name="Mystic RabbitMQ"},
        @{Port=15672; Name="Mystic RabbitMQ Management"}
    )
    
    # Monitoring Ports
    $monitoringPorts = @(
        @{Port=9090; Name="Mystic Prometheus"},
        @{Port=3000; Name="Mystic Grafana"}
    )
    
    # Advanced Feature Ports
    $advancedPorts = @(
        @{Port=8087; Name="Mystic Quantum Trading Engine"},
        @{Port=8088; Name="Mystic Quantum Optimization"},
        @{Port=8089; Name="Mystic Quantum Machine Learning"},
        @{Port=8106; Name="Mystic QKD Alice"},
        @{Port=8093; Name="Mystic 5G Core"},
        @{Port=8094; Name="Mystic 5G RAN"},
        @{Port=8095; Name="Mystic 5G Slice Manager"},
        @{Port=8099; Name="Mystic Bitcoin Miner"},
        @{Port=8100; Name="Mystic Ethereum Miner"},
        @{Port=8101; Name="Mystic Mining Pool"},
        @{Port=8096; Name="Mystic Satellite Receiver"},
        @{Port=8097; Name="Mystic Satellite Processor"},
        @{Port=8098; Name="Mystic Satellite Analytics"},
        @{Port=8090; Name="Mystic Edge Node 1"},
        @{Port=8091; Name="Mystic Edge Node 2"},
        @{Port=8092; Name="Mystic Edge Orchestrator"},
        @{Port=8102; Name="Mystic AI Super Master"},
        @{Port=2181; Name="Mystic Zookeeper"},
        @{Port=8265; Name="Mystic Ray Dashboard"},
        @{Port=10001; Name="Mystic Ray Port"}
    )
    
    $allPorts = $corePorts + $aiPorts + $cachePorts + $queuePorts + $monitoringPorts + $advancedPorts
    
    foreach ($portConfig in $allPorts) {
        $port = $portConfig.Port
        $name = $portConfig.Name
        
        try {
            # Remove existing rule if it exists
            Remove-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
            
            # Add new rule
            New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP -LocalPort $port -Action Allow -Profile Any
            
            Write-ColorOutput "✅ Added firewall rule for $name (Port $port)" Green
        }
        catch {
            Write-ColorOutput "❌ Failed to add firewall rule for $name (Port $port): $($_.Exception.Message)" Red
        }
    }
    
    Write-ColorOutput "🎯 Firewall configuration completed!" Green
}

function Remove-MysticFirewallRules {
    Write-ColorOutput "🗑️ Removing Mystic Trading Platform Firewall Rules..." Yellow
    
    $ruleNames = @(
        "Mystic Backend", "Mystic Dashboard", "Mystic API", "Mystic Trade Logging",
        "Mystic AI Service", "Mystic AI Processor", "Mystic Visualization", "Mystic AI Trade Engine",
        "Mystic Redis", "Mystic Redis Cluster 1", "Mystic Redis Cluster 2", "Mystic Redis Cluster 3",
        "Mystic RabbitMQ", "Mystic RabbitMQ Management",
        "Mystic Prometheus", "Mystic Grafana",
        "Mystic Quantum Trading Engine", "Mystic Quantum Optimization", "Mystic Quantum Machine Learning", "Mystic QKD Alice",
        "Mystic 5G Core", "Mystic 5G RAN", "Mystic 5G Slice Manager",
        "Mystic Bitcoin Miner", "Mystic Ethereum Miner", "Mystic Mining Pool",
        "Mystic Satellite Receiver", "Mystic Satellite Processor", "Mystic Satellite Analytics",
        "Mystic Edge Node 1", "Mystic Edge Node 2", "Mystic Edge Orchestrator",
        "Mystic AI Super Master", "Mystic Zookeeper", "Mystic Ray Dashboard", "Mystic Ray Port"
    )
    
    foreach ($ruleName in $ruleNames) {
        try {
            Remove-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
            Write-ColorOutput "✅ Removed firewall rule: $ruleName" Green
        }
        catch {
            Write-ColorOutput "ℹ️ Rule not found: $ruleName" Yellow
        }
    }
    
    Write-ColorOutput "🗑️ Firewall rules removal completed!" Green
}

function Show-MysticFirewallRules {
    Write-ColorOutput "📋 Current Mystic Trading Platform Firewall Rules:" Cyan
    
    $mysticRules = Get-NetFirewallRule | Where-Object { $_.DisplayName -like "Mystic*" }
    
    if ($mysticRules) {
        $mysticRules | Format-Table DisplayName, Direction, Protocol, LocalPort, Action, Enabled -AutoSize
    } else {
        Write-ColorOutput "ℹ️ No Mystic firewall rules found." Yellow
    }
}

function Test-MysticPorts {
    Write-ColorOutput "🔍 Testing Mystic Trading Platform Ports..." Cyan
    
    $ports = @(9000, 8501, 8000, 8080, 8001, 8002, 8003, 8004, 6379, 6380, 6381, 6382, 5672, 15672, 9090, 3000)
    
    foreach ($port in $ports) {
        try {
            $connection = Test-NetConnection -ComputerName localhost -Port $port -InformationLevel Quiet
            if ($connection.TcpTestSucceeded) {
                Write-ColorOutput "✅ Port $port is OPEN" Green
            } else {
                Write-ColorOutput "❌ Port $port is CLOSED" Red
            }
        }
        catch {
            Write-ColorOutput "❌ Port $port test failed" Red
        }
    }
}

# Main execution
if (-not (Test-Administrator)) {
    Write-ColorOutput "❌ This script must be run as Administrator!" Red
    Write-ColorOutput "Please right-click PowerShell and select 'Run as Administrator'" Yellow
    exit 1
}

Write-ColorOutput "🚀 Mystic Trading Platform Firewall Configuration" Cyan
Write-ColorOutput "=================================================" Cyan

if ($RemoveRules) {
    Remove-MysticFirewallRules
}
elseif ($ListRules) {
    Show-MysticFirewallRules
}
elseif ($TestPorts) {
    Test-MysticPorts
}
else {
    Add-MysticFirewallRules
    
    Write-ColorOutput "" White
    Write-ColorOutput "📋 Next Steps:" Cyan
    Write-ColorOutput "1. Start Redis: .\scripts\start-redis.bat" White
    Write-ColorOutput "2. Start Backend: .\scripts\start-backend.bat" White
    Write-ColorOutput "3. Start Dashboard: .\scripts\start-frontend.bat" White
    Write-ColorOutput "4. Test: curl http://localhost:9000/health" White
    Write-ColorOutput "5. Open Dashboard: http://localhost:8501" White
    
    Write-ColorOutput "" White
    Write-ColorOutput "🔧 Usage Options:" Cyan
    Write-ColorOutput "  .\scripts\setup-firewall.ps1 -RemoveRules  # Remove all rules" White
    Write-ColorOutput "  .\scripts\setup-firewall.ps1 -ListRules    # Show current rules" White
    Write-ColorOutput "  .\scripts\setup-firewall.ps1 -TestPorts    # Test port availability" White
} 