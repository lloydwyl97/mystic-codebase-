Write-Host "🧹 Cleaning Vite cache and rebuilding project..." -ForegroundColor Green
Write-Host ""

# Remove .vite directory
if (Test-Path ".vite") {
    Write-Host "Removing .vite directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".vite"
}

# Remove node_modules
if (Test-Path "node_modules") {
    Write-Host "Removing node_modules..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "node_modules"
}

# Remove package-lock.json
if (Test-Path "package-lock.json") {
    Write-Host "Removing package-lock.json..." -ForegroundColor Yellow
    Remove-Item "package-lock.json"
}

# Clean npm cache
Write-Host "Cleaning npm cache..." -ForegroundColor Yellow
npm cache clean --force

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
npm install

# Start development server
Write-Host "Starting development server on port 3000..." -ForegroundColor Green
npm run dev -- --port 3000 