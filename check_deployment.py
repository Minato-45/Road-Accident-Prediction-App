#!/usr/bin/env python3
"""
Deployment readiness checker for Road Accident Prediction app
This script verifies that all necessary files are present for Render deployment.
"""

import os
import sys

def check_file_exists(file_path, description):
    """Check if a file exists and print status."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False

def main():
    print("🚀 Render Deployment Readiness Check")
    print("=" * 50)
    
    # Core application files
    core_files = [
        ("app.py", "Main Flask application"),
        ("requirements.txt", "Python dependencies"),
        ("Procfile", "Render process configuration"),
        ("runtime.txt", "Python runtime version"),
        ("render.yaml", "Render deployment config"),
        ("model.pkl", "Machine learning model"),
        ("README.md", "Project documentation"),
        (".gitignore", "Git ignore rules")
    ]
    
    # Essential directories
    directories = [
        ("templates", "HTML templates"),
        ("static", "Static files (CSS, JS, images)"),
        ("resources", "Python data modules")
    ]
    
    # Check core files
    print("\n📋 Core Files:")
    missing_files = 0
    for file_path, description in core_files:
        if not check_file_exists(file_path, description):
            missing_files += 1
    
    # Check directories
    print("\n📁 Essential Directories:")
    missing_dirs = 0
    for dir_path, description in directories:
        if os.path.isdir(dir_path):
            print(f"✅ {description}: {dir_path}/")
        else:
            print(f"❌ {description}: {dir_path}/ (MISSING)")
            missing_dirs += 1
    
    # Check important subdirectories and files
    print("\n🎨 Static Assets:")
    static_items = [
        ("static/css", "CSS files directory"),
        ("static/js", "JavaScript files directory"),
        ("static/images", "Images directory"),
        ("resources/data.py", "Data constants module")
    ]
    
    for item_path, description in static_items:
        if os.path.exists(item_path):
            print(f"✅ {description}: {item_path}")
        else:
            print(f"❌ {description}: {item_path} (MISSING)")
            missing_files += 1
    
    # Summary
    print("\n" + "=" * 50)
    total_missing = missing_files + missing_dirs
    
    if total_missing == 0:
        print("🎉 SUCCESS! Your project is ready for Render deployment.")
        print("\nNext steps:")
        print("1. Create a GitHub repository")
        print("2. Push your code to GitHub")
        print("3. Deploy on Render using the GitHub repository")
        print("4. Follow the DEPLOYMENT_GUIDE.md for detailed instructions")
    else:
        print(f"⚠️  WARNING: {total_missing} items are missing.")
        print("Please ensure all required files and directories are present before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()