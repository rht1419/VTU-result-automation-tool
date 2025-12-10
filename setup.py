#!/usr/bin/env python3
"""
Setup script for VTU Results Automation System
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python 3.7+ is installed"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        return False
    return True

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to install requirements.")
        return False

def install_playwright_browsers():
    """Install Playwright browsers"""
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install"])
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to install Playwright browsers.")
        return False

def main():
    print("VTU Results Automation System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    print("Installing Python requirements...")
    if not install_requirements():
        sys.exit(1)
    
    # Install Playwright browsers
    print("Installing Playwright browsers...")
    if not install_playwright_browsers():
        sys.exit(1)
    
    print("\nSetup completed successfully!")
    print("\nTo start the web interface, run:")
    print("  python vtu_ultra_fast_web_interface.py")
    print("\nThen open your browser to http://localhost:5001")

if __name__ == "__main__":
    main()