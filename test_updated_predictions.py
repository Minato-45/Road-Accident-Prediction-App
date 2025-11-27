"""
Test the updated prediction system with same inputs as before
"""
import sys
sys.path.append('.')

# Import the updated app components
import resources.data as data
from app import robust_feature_mapping
import pickle
import pandas as pd

print("🧪 Testing updated prediction system...")

# Load the regenerated model
print("📦 Loading regenerated model...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"✅ Model loaded: {type(model).__name__}")

# Test with the same features as before
test_features = [
    'Karnataka',        # State
    'T-Junction',       # Junction  
    'Less than 5 years', # Vehicle age
    '18 Yrs -Male',     # Human age/sex
    'Drivers',          # Safety precautions (PERSON column)
    'Residential Area', # Area
    'Urban',            # Type of place
    'Normally Loaded',  # Vehicle load
    'Over-Speeding',    # Traffic violation
    'Sunny/Clear',      # Weather
    'Pedestrian - Male', # Vehicle type/sex
    'Straight Road',    # Road type
    'License Valid Permanent', # License
    '06-0900hrs - (Day)' # Time
]

print(f"\n🧪 Testing with features: {test_features}")

# Create test data structure (same as app.py predict function)
testData = {
    'States/UTs':[test_features[0]], 
    'JUNCTION':[test_features[1]], 
    'VEHICLE AGE':[test_features[2]],
    'HUMAN AGE AND SEX':[test_features[3]], 
    'PERSON WITHOUT SAFETY PRECAUTIONS':[test_features[4]],  # This maps to PERSON column
    'AREA':[test_features[5]], 
    'TYPE OF PLACE':[test_features[6]], 
    'LOAD OF VEHICLE':[test_features[7]],
    'TRAFFIC RULES VIOLATION':[test_features[8]], 
    'WEATHER':[test_features[9]], 
    'VEHICLE TYPE AND SEX':[test_features[10]], 
    'TYPE OF ROAD':[test_features[11]], 
    'LICENSE':[test_features[12]],
    'TIME':[test_features[13]]
}

# Convert to codes using robust mapping
print("\n🔢 Converting to codes using robust mapping:")
for i, col in enumerate(testData):
    feature_value = ''.join(testData[col])
    code = robust_feature_mapping(feature_value, col)
    testData[col] = [code]
    print(f"   {col}: '{feature_value}' -> {code}")

# Create DataFrame and predict
testDataFrame = pd.DataFrame.from_dict(testData)
print(f"\n📊 DataFrame shape: {testDataFrame.shape}")
print(f"📊 DataFrame values: {testDataFrame.values}")

# Make prediction
prediction = model.predict(testDataFrame)
try:
    probabilities = model.predict_proba(testDataFrame)
    print(f"🔮 Prediction: {prediction[0]} (Probabilities: {probabilities[0]})")
except:
    print(f"🔮 Prediction: {prediction[0]}")

if prediction[0] == 0:
    result = "No, There is No Chance of Road Accident."
else:
    result = "Yes, There is a Chance Of Road Accident! Be Careful."
    
print(f"📝 Final result: {result}")

print("\n🧪 Testing multiple scenarios for variety...")

# Test multiple scenarios
scenarios = [
    # High risk scenario
    ['Uttar Pradesh', 'Four arm Junction', '> 15 Years', '18-25 Yrs - Male', 'Drivers', 
     'Market/Commercial Area', 'Urban', 'Overloaded/Hangin', 'Over-Speeding', 'Foggy & Misty',
     'Two Wheelers - Male', 'Curved Road', 'Without Licence', '21-2400hrs - (Night)'],
    
    # Low risk scenario
    ['Kerala', 'Others', 'Less than 5 years', '35-40 Yrs - Female', 'Passengers',
     'Residential Area', 'Rural', 'Normally Loaded', 'Driving on Wrong Side', 'Sunny/Clear',
     'Cars & taxies Vans & LMV - Female', 'Straight Road', 'License Valid Permanent', '09-1200hrs - (Day)'],
     
    # Medium risk scenario
    ['Maharashtra', 'T-Junction', '5.1 - 10 Years', '25-35 Yrs- Male', 'Drivers',
     'Open Area', 'Urban', 'Normally Loaded', 'Use of Mobile Phone', 'Rainy',
     'Auto Rickshaws - Male', 'Bridge', 'License Valid Permanent', '15-1800hrs - (Day)']
]

for i, scenario in enumerate(scenarios, 1):
    testData_scenario = {
        'States/UTs':[scenario[0]], 'JUNCTION':[scenario[1]], 'VEHICLE AGE':[scenario[2]],
        'HUMAN AGE AND SEX':[scenario[3]], 'PERSON WITHOUT SAFETY PRECAUTIONS':[scenario[4]],
        'AREA':[scenario[5]], 'TYPE OF PLACE':[scenario[6]], 'LOAD OF VEHICLE':[scenario[7]],
        'TRAFFIC RULES VIOLATION':[scenario[8]], 'WEATHER':[scenario[9]], 
        'VEHICLE TYPE AND SEX':[scenario[10]], 'TYPE OF ROAD':[scenario[11]], 
        'LICENSE':[scenario[12]], 'TIME':[scenario[13]]
    }
    
    # Encode
    for col in testData_scenario:
        feature_value = ''.join(testData_scenario[col])
        code = robust_feature_mapping(feature_value, col)
        testData_scenario[col] = [code]
    
    # Predict
    df_scenario = pd.DataFrame.from_dict(testData_scenario)
    pred = model.predict(df_scenario)[0]
    prob = model.predict_proba(df_scenario)[0]
    
    result_text = "Accident Risk" if pred == 1 else "No Accident"
    print(f"Scenario {i}: {result_text} (confidence: {prob[pred]:.3f})")

print("\n✅ Prediction testing completed!")