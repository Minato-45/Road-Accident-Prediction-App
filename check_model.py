"""
Check the current model type and test prediction consistency
"""
import pickle
import pandas as pd
import resources.data as data

# Load the model
print("Loading model...")
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"   Model params: {model.get_params()}")
    
    # Test with a sample prediction
    test_features = [
        'Karnataka',        # State
        'T Junction',       # Junction  
        '5-10 years',       # Vehicle age
        'Male 25-35',       # Human age/sex
        'No',               # Safety precautions
        'Urban',            # Area
        'Market',           # Type of place
        'Overloaded',       # Vehicle load
        'Over Speeding',    # Traffic violation
        'Clear',            # Weather
        'Two Wheeler Male', # Vehicle type/sex
        'State Highway',    # Road type
        'Driving License',  # License
        'Day'               # Time
    ]
    
    print(f"\n🧪 Testing with features: {test_features}")
    
    # Create test data structure
    testData = {
        'States/UTs':[test_features[0]], 
        'JUNCTION':[test_features[1]], 
        'VEHICLE AGE':[test_features[2]],
        'HUMAN AGE AND SEX':[test_features[3]], 
        'PERSON WITHOUT SAFETY PRECAUTIONS':[test_features[4]],
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
    
    # Convert to codes
    print("\n🔢 Converting to numerical codes:")
    for col in testData:
        feature_value = ''.join(testData[col])
        if feature_value in data.columnCodes:
            code = [data.columnCodes[feature_value]]
            testData[col] = code
            print(f"   {col}: '{feature_value}' -> {code[0]}")
        else:
            print(f"   ❌ UNKNOWN: {col}: '{feature_value}' -> using 0")
            testData[col] = [0]
    
    # Create DataFrame and predict
    testDataFrame = pd.DataFrame.from_dict(testData)
    print(f"\n📊 DataFrame shape: {testDataFrame.shape}")
    print(f"📊 DataFrame values: {testDataFrame.values}")
    
    # Make prediction
    prediction = model.predict(testDataFrame)
    probabilities = None
    try:
        probabilities = model.predict_proba(testDataFrame)
        print(f"🔮 Prediction: {prediction[0]} (Probabilities: {probabilities[0]})")
    except:
        print(f"🔮 Prediction: {prediction[0]}")
    
    if prediction[0] == 0:
        result = "No Accident"
    else:
        result = "Accident Risk"
        
    print(f"📝 Final result: {result}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()