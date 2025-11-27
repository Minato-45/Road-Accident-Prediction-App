#!/usr/bin/env python3
"""
Test the new model to ensure it gives varied predictions
"""

import pickle
import numpy as np
import pandas as pd
from resources import data

def test_model_predictions():
    """Test the model with various scenarios"""
    
    print("🧪 Testing Model Predictions")
    print("=" * 40)
    
    # Load the model
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test scenarios - create different combinations that should give different results
    test_scenarios = [
        {
            'name': 'High Risk Scenario',
            'features': ['Andhra Pradesh', 'T-Junction', 'Less than 5 years', '18 Yrs -Male', 
                        'Drivers', 'Market/Commercial Area', 'Urban', 'Overloaded/Hangin', 
                        'Drunken Driving/Consumption of Alcohol & Drug', 'Rainy', 
                        'Two Wheelers - Male', 'Curved Road', 'Without Licence', '21-2400hrs - (Night)']
        },
        {
            'name': 'Medium Risk Scenario',
            'features': ['Karnataka', 'Y-Junction', '5.1 - 10 Years', '25-35 Yrs- Male',
                        'Drivers', 'Residential Area', 'Urban', 'Normally Loaded',
                        'Over-Speeding', 'Sunny/Clear', 'Cars & taxies Vans & LMV - Male', 
                        'Straight Road', 'License Valid Permanent', '12-1500hrs - (Day)']
        },
        {
            'name': 'Low Risk Scenario', 
            'features': ['Himachal Pradesh', 'Round about Junction', '> 15 Years', '45-60 Yrs- Male',
                        'Drivers', 'Residential Area', 'Rural', 'Normally Loaded',
                        'Over-Speeding', 'Sunny/Clear', 'Cars & taxies Vans & LMV - Male',
                        'Straight Road', 'License Valid Permanent', '09-1200hrs - (Day)']
        },
        {
            'name': 'Another Test Scenario',
            'features': ['Delhi', 'Four arm Junction', 'Less than 5 years', '18-25 Yrs - Female',
                        'Passengers', 'Market/Commercial Area', 'Urban', 'Others',
                        'Use of Mobile Phone', 'Foggy & Misty', 'Pedestrian - Female',
                        'Ongoing Road Works/Under Construction', 'Learner\'s Licence', '18-2100hrs - (Night)']
        },
        {
            'name': 'Safe Driving Scenario',
            'features': ['Goa', 'Round about Junction', '5.1 - 10 Years', '35-40 Yrs- Male',
                        'Drivers', 'Residential Area', 'Rural', 'Normally Loaded',
                        'Over-Speeding', 'Sunny/Clear', 'Cars & taxies Vans & LMV - Male',
                        'Straight Road', 'License Valid Permanent', '06-0900hrs - (Day)']
        }
    ]
    
    predictions = []
    
    for scenario in test_scenarios:
        print(f"\n🔍 Testing: {scenario['name']}")
        
        try:
            # Encode features using the same logic as app.py
            userFeatures = scenario['features']
            
            # Create test data dictionary
            testData = {
                'States/UTs': [userFeatures[0]], 
                'JUNCTION': [userFeatures[1]], 
                'VEHICLE AGE': [userFeatures[2]],
                'HUMAN AGE AND SEX': [userFeatures[3]], 
                'PERSON WITHOUT SAFETY PRECAUTIONS': [userFeatures[4]],
                'AREA': [userFeatures[5]], 
                'TYPE OF PLACE': [userFeatures[6]], 
                'LOAD OF VEHICLE': [userFeatures[7]],
                'TRAFFIC RULES VIOLATION': [userFeatures[8]], 
                'WEATHER': [userFeatures[9]], 
                'VEHICLE TYPE AND SEX': [userFeatures[10]], 
                'TYPE OF ROAD': [userFeatures[11]], 
                'LICENSE': [userFeatures[12]],
                'TIME': [userFeatures[13]]
            }
            
            # Convert to codes
            for col in testData:
                feature_value = ''.join(testData[col])
                if feature_value in data.columnCodes:
                    code = [data.columnCodes[feature_value]]
                    testData[col] = code
                else:
                    print(f"   ⚠️ Unknown feature value: {feature_value}")
                    testData[col] = [0]
            
            testDataFrame = pd.DataFrame.from_dict(testData)
            
            # Make prediction
            prediction = model.predict(testDataFrame)[0]
            
            # Get probability if available
            try:
                probabilities = model.predict_proba(testDataFrame)[0]
                confidence = max(probabilities) * 100
                prob_text = f"(Confidence: {confidence:.1f}%)"
            except:
                prob_text = ""
            
            result = "🚨 ACCIDENT RISK" if prediction == 1 else "✅ NO ACCIDENT"
            print(f"   Prediction: {result} {prob_text}")
            
            predictions.append(prediction)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            predictions.append(-1)
    
    # Summary
    print(f"\n" + "=" * 40)
    print("📊 PREDICTION SUMMARY")
    print("=" * 40)
    
    accident_count = sum(1 for p in predictions if p == 1)
    no_accident_count = sum(1 for p in predictions if p == 0)
    error_count = sum(1 for p in predictions if p == -1)
    
    print(f"🚨 Accident predictions: {accident_count}")
    print(f"✅ No accident predictions: {no_accident_count}")
    print(f"❌ Errors: {error_count}")
    
    if accident_count > 0 and no_accident_count > 0:
        print("🎉 SUCCESS: Model gives varied predictions!")
    elif accident_count == 0:
        print("⚠️  WARNING: Model never predicts accidents")
    elif no_accident_count == 0:
        print("⚠️  WARNING: Model always predicts accidents")
    
    return accident_count > 0 and no_accident_count > 0

if __name__ == "__main__":
    test_model_predictions()