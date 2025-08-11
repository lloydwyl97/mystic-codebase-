# ===== DOCKER CLEANUP SCRIPT - REMOVE BLOAT =====

Write-Host "🧹 Cleaning up Docker bloat..." -ForegroundColor Green

# Stop all containers
Write-Host "🛑 Stopping all containers..." -ForegroundColor Yellow
docker-compose down

# Remove all mystic images (the bloated ones)
Write-Host "🗑️ Removing bloated images..." -ForegroundColor Yellow
docker images --format "{{.Repository}}:{{.Tag}}" | findstr "mystic-codebase" | ForEach-Object {
    Write-Host "  Removing $_" -ForegroundColor Red
    docker rmi $_
}

# Remove dangling images
Write-Host "🧹 Removing dangling images..." -ForegroundColor Yellow
docker image prune -f

# Remove build cache
Write-Host "🗑️ Removing build cache..." -ForegroundColor Yellow
docker builder prune -a -f

# Show space reclaimed
Write-Host "📊 Docker system status after cleanup:" -ForegroundColor Green
docker system df

Write-Host "✅ Cleanup completed! Ready for optimized rebuild." -ForegroundColor Green 