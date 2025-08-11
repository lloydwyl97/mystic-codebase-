#!/usr/bin/env python3
"""
Python Interpreter Configuration Helper
Use this to set the correct Python interpreter in your IDE
"""

import sys
import os

def main():
    print("=" * 60)
    print("PYTHON INTERPRETER CONFIGURATION")
    print("=" * 60)
    
    # Get current Python executable
    python_path = sys.executable
    print(f"✅ Current Python Path: {python_path}")
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit is available in this environment")
    except ImportError:
        print("❌ Streamlit is NOT available in this environment")
    
    # Check other key packages
    packages_to_check = [
        'pandas', 'numpy', 'plotly', 'matplotlib', 'seaborn',
        'requests', 'aiohttp', 'fastapi', 'uvicorn'
    ]
    
    print("\n📦 Package Availability Check:")
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS FOR CURSOR IDE:")
    print("=" * 60)
    print("1. Press Ctrl+Shift+P to open Command Palette")
    print("2. Type 'Python: Select Interpreter' and select it")
    print("3. Choose this path:")
    print(f"   {python_path}")
    print("4. Restart Cursor if needed")
    print("=" * 60)

if __name__ == "__main__":
    main()

