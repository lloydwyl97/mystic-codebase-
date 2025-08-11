#!/bin/bash

# Caddy Installation Script for Mystic Trading Platform
echo "🚀 Installing Caddy Web Server..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        echo "📦 Installing Caddy on Ubuntu/Debian..."
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
        sudo apt update
        sudo apt install caddy
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        echo "📦 Installing Caddy on CentOS/RHEL..."
        dnf install 'dnf-command(copr)'
        dnf copr enable @caddy/caddy
        dnf install caddy
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        echo "📦 Installing Caddy on Arch Linux..."
        sudo pacman -S caddy
    else
        echo "❌ Unsupported Linux distribution"
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "📦 Installing Caddy on macOS..."
    if command -v brew &> /dev/null; then
        brew install caddy
    else
        echo "❌ Please install Homebrew first: https://brew.sh"
        exit 1
    fi
else
    echo "❌ Unsupported operating system"
    exit 1
fi

# Verify installation
if command -v caddy &> /dev/null; then
    echo "✅ Caddy installed successfully!"
    caddy version
else
    echo "❌ Caddy installation failed"
    exit 1
fi
