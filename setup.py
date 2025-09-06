#!/usr/bin/env python3
import os
import subprocess
import sys

def install_requirements():
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed successfully!")

def setup_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        print("✅ GOOGLE_API_KEY is already set.")
    else:
        print("\n🚨 GOOGLE_API_KEY is not set.")
        print("Please run:")
        print("export GOOGLE_API_KEY='AIzaSyD5kTuCzae-t7M5ZoiIpOhps-zh_dNbTWc'")
        print("Or add to your shell profile (~/.zshrc, ~/.bashrc).")

def main():
    print("🚀 Setting up Tresata Classifier...")
    install_requirements()
    setup_api_key()
    print("\nUsage:")
    print("python classifier.py --input sample.csv --column data")

if __name__ == "__main__":
    main()
