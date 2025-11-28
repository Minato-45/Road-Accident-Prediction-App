#!/usr/bin/env python3
"""
Database Test Script for Road Accident Prediction App
This script tests the user database functionality
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_database_status():
    """Test the database status endpoint"""
    try:
        print("📊 Testing database status...")
        response = requests.get(f"{BASE_URL}/db-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Database Status: {data.get('database_status')}")
            print(f"📁 File Path: {data.get('file_path')}")
            print(f"👥 User Count: {data.get('user_count')}")
            print(f"💾 File Exists: {data.get('file_exists')}")
            print(f"🔓 File Readable: {data.get('file_readable')}")
            print(f"🔒 File Writable: {data.get('file_writable')}")
            return True
        else:
            print(f"❌ Database status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking database status: {e}")
        return False

def test_user_list():
    """Test the user listing endpoint"""
    try:
        print("\n👥 Testing user listing...")
        response = requests.get(f"{BASE_URL}/users", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Total Users: {data.get('total_users')}")
            if data.get('users'):
                for user in data['users']:
                    print(f"   🧑 {user['username']} ({user['fullname']}) - Created: {user['created_at']}")
            return True
        else:
            print(f"❌ User list check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking user list: {e}")
        return False

def test_registration_and_login():
    """Test registration and login functionality"""
    try:
        print("\n🔑 Testing registration and login...")
        
        # Test data
        test_user = {
            'fullname': 'Test User Database',
            'email': f'testdb{int(time.time())}@example.com',
            'username': f'testdb_{int(time.time())}',
            'password': 'testpass123',
            'confirm_password': 'testpass123'
        }
        
        # Test registration
        print("📝 Testing registration...")
        session = requests.Session()
        
        # Get the registration page first (to simulate real usage)
        session.get(f"{BASE_URL}/register")
        
        # Attempt registration
        reg_response = session.post(f"{BASE_URL}/register", data=test_user, allow_redirects=False)
        
        if reg_response.status_code == 302:
            print(f"✅ Registration successful for {test_user['username']}")
            
            # Test login
            print("🔐 Testing login...")
            login_data = {
                'username': test_user['username'],
                'password': test_user['password']
            }
            
            login_response = session.post(f"{BASE_URL}/login", data=login_data, allow_redirects=False)
            
            if login_response.status_code == 302:
                print(f"✅ Login successful for {test_user['username']}")
                return True
            else:
                print(f"❌ Login failed: {login_response.status_code}")
                return False
        else:
            print(f"❌ Registration failed: {reg_response.status_code}")
            print(f"Response text: {reg_response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Error during registration/login test: {e}")
        return False

def main():
    """Run all database tests"""
    print("🧪 Road Accident App Database Tests")
    print("=" * 50)
    
    tests = [
        test_database_status,
        test_user_list,
        test_registration_and_login
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All database tests passed! Your database is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    main()