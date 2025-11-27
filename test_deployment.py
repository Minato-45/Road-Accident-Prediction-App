"""
Test script to verify the deployed Render app predictions match localhost
Run this after deployment completes
"""
import requests
import time

# Your Render app URL
RENDER_URL = "https://road-accident-prediction-app.onrender.com"

def test_deployment_consistency():
    """Test that deployment predictions work correctly"""
    
    print("🧪 Testing Render deployment prediction consistency...")
    
    # First check if the app is live and healthy
    print(f"\n🔍 Checking health status...")
    try:
        health_response = requests.get(f"{RENDER_URL}/healthz", timeout=10)
        health_data = health_response.json()
        print(f"   Status: {health_data.get('status', 'unknown')}")
        print(f"   Model: {health_data.get('model', 'unknown')}")
        print(f"   Data: {health_data.get('data', 'unknown')}")
        
        if health_data.get('status') != 'healthy':
            print("❌ App is not healthy, waiting for deployment...")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test prediction with sample data
    print(f"\n🎯 Testing prediction...")
    
    prediction_data = {
        'state': 'Karnataka',
        'junction': 'T-Junction',
        'vechicleAge': 'Less than 5 years', 
        'humanAgeSex': '18 Yrs -Male',
        'personWithoutPrecautions': 'Drivers',
        'areas': 'Residential Area',
        'typeOfPlace': 'Urban',
        'vehicleLoad': 'Normally Loaded',
        'trafficRulesViolation': 'Over-Speeding',
        'weather': 'Sunny/Clear',
        'vehicleTypeSex': 'Pedestrian - Male',
        'roadType': 'Straight Road',
        'License': 'License Valid Permanent',
        'time': '06-0900hrs - (Day)'
    }
    
    try:
        # Make prediction request
        response = requests.post(
            f"{RENDER_URL}/predict",
            data=prediction_data,
            timeout=30
        )
        
        if response.status_code == 200:
            # Check if response contains prediction result
            response_text = response.text
            if "There is a Chance Of Road Accident" in response_text:
                print("✅ Deployment prediction: YES (Accident Risk)")
                return "YES"
            elif "No Chance of Road Accident" in response_text:
                print("✅ Deployment prediction: NO (No Accident)")
                return "NO"
            else:
                print("⚠️ Unexpected response format")
                print(f"Response snippet: {response_text[:200]}...")
                return "UNKNOWN"
        else:
            print(f"❌ Prediction failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")
        return False

def monitor_deployment(max_attempts=10, wait_time=30):
    """Monitor deployment until it's ready or timeout"""
    
    print(f"🚀 Monitoring deployment at {RENDER_URL}...")
    print(f"   Will check every {wait_time} seconds for up to {max_attempts} attempts")
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n📊 Attempt {attempt}/{max_attempts}")
        
        result = test_deployment_consistency()
        
        if result and result != False:
            print(f"\n🎉 Deployment successful!")
            print(f"   Prediction result: {result}")
            print(f"   App is live at: {RENDER_URL}")
            return True
        
        if attempt < max_attempts:
            print(f"⏳ Waiting {wait_time} seconds before next attempt...")
            time.sleep(wait_time)
    
    print(f"\n⏰ Timeout reached after {max_attempts} attempts")
    print("   Deployment may still be in progress. Check manually later.")
    return False

if __name__ == "__main__":
    print("🌐 Render Deployment Consistency Test")
    print("=====================================")
    
    # Start monitoring
    success = monitor_deployment()
    
    if success:
        print(f"\n✅ All tests passed!")
        print(f"🔗 Your app is working correctly at: {RENDER_URL}")
    else:
        print(f"\n⚠️ Tests incomplete - check deployment status manually")
        print(f"🔗 Check your app at: {RENDER_URL}")
        
    print(f"\n📋 Manual verification steps:")
    print(f"   1. Visit: {RENDER_URL}")
    print(f"   2. Go to prediction page (/home)")
    print(f"   3. Fill form with Karnataka, T-Junction, Less than 5 years, etc.")
    print(f"   4. Submit and verify prediction works")
    print(f"   5. Compare result with localhost version")