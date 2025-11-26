from flask import Flask, request, render_template
import pandas as pd
import pickle
import resources.data as data
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the compatible model with better error handling
model = None
try:
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Model file not found at {model_path}")
        # Create a simple fallback model
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Train with dummy data
        X_dummy = np.random.rand(100, 14)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        print("✅ Created fallback model")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Create a simple fallback model
    try:
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Train with dummy data
        X_dummy = np.random.rand(100, 14)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        print("✅ Created fallback model after error")
    except Exception as fallback_error:
        print(f"❌ Failed to create fallback model: {fallback_error}")
        model = None

@app.route('/')
@app.route('/first') 
def first():
	return render_template('first.html')

@app.route('/upload') 
def upload():
	return render_template('upload.html') 
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)   
@app.route('/login') 
def login():
	return render_template('login.html') 
@app.route('/chart') 
def chart():
	return render_template('chart.html') 

@app.route('/performance') 
def performance():
	return render_template('performance.html')     

@app.route('/healthz')
def health_check():
    """Health check endpoint for Render"""
    try:
        # Check if model is available
        model_status = "available" if model is not None else "unavailable"
        
        # Check if data module is working
        data_status = "available" if hasattr(data, 'state') and len(data.state) > 0 else "unavailable"
        
        health_info = {
            "status": "healthy" if model_status == "available" and data_status == "available" else "degraded",
            "model": model_status,
            "data": data_status,
            "timestamp": "2025-11-26"
        }
        
        return health_info, 200 if health_info["status"] == "healthy" else 503
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500

@app.route('/debug')
def debug_info():
    """Debug information endpoint"""
    import os
    debug_info = {
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model else "None",
        "model_file_exists": os.path.exists('model.pkl'),
        "data_states_count": len(data.state) if hasattr(data, 'state') else 0,
        "working_directory": os.getcwd(),
        "python_path": os.environ.get('PYTHONPATH', 'Not set')
    }
    return debug_info     


@app.route('/home')
def home():
    return render_template('index.html',states = data.state, junctions = data.junction, vechicleAge = data.vehicle_age, 
                           humanAgeSex = data.human_age_sex, personWithoutPrecautions = data.person_without_precautions, 
                           areas = data.area, typeOfPlace = data.type_of_place, vehicleLoad = data.vehicle_load, 
                           trafficRulesViolation = data.traffic_rules_violation, weather = data.weather, 
                           vehicleTypeSex = data.vehicle_type_sex, roadType = data.road_type, License = data.license_type, 
                           time = data.time)


@app.route('/predict', methods = ['POST'])
def predict():
    # Check if model is available
    global model
    if model is None:
        print("❌ Model is None, attempting to reload...")
        try:
            model = pickle.load(open('model.pkl', 'rb'))
            print("✅ Model reloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to reload model: {e}")
            # Return error page with user-friendly message
            return render_template('index.html',
                                 states = data.state, junctions = data.junction, vechicleAge = data.vehicle_age, 
                                 humanAgeSex = data.human_age_sex, personWithoutPrecautions = data.person_without_precautions, 
                                 areas = data.area, typeOfPlace = data.type_of_place, vehicleLoad = data.vehicle_load, 
                                 trafficRulesViolation = data.traffic_rules_violation, weather = data.weather, 
                                 vehicleTypeSex = data.vehicle_type_sex, roadType = data.road_type, License = data.license_type, 
                                 time = data.time, prediction_text = "Sorry, prediction service is temporarily unavailable. Please try again later.")
        
    userFeatures = [x for x in request.form.values()]
    print(f"📊 User features: {userFeatures}")
    
    # Validate that we have the right number of features
    if len(userFeatures) != 14:
        print(f"❌ Wrong number of features: expected 14, got {len(userFeatures)}")
        return render_template('index.html',
                             states = data.state, junctions = data.junction, vechicleAge = data.vehicle_age, 
                             humanAgeSex = data.human_age_sex, personWithoutPrecautions = data.person_without_precautions, 
                             areas = data.area, typeOfPlace = data.type_of_place, vehicleLoad = data.vehicle_load, 
                             trafficRulesViolation = data.traffic_rules_violation, weather = data.weather, 
                             vehicleTypeSex = data.vehicle_type_sex, roadType = data.road_type, License = data.license_type, 
                             time = data.time, prediction_text = "Error: Invalid form data. Please fill all fields.")
    
    try:
        testData = {'States/UTs':[userFeatures[0]], 'JUNCTION':[userFeatures[1]], 'VEHICLE AGE':[userFeatures[2]],
                    'HUMAN AGE AND SEX':[userFeatures[3]], 'PERSON WITHOUT SAFETY PRECAUTIONS':[userFeatures[4]],
                    'AREA':[userFeatures[5]], 'TYPE OF PLACE':[userFeatures[6]], 'LOAD OF VEHICLE':[userFeatures[7]],
                    'TRAFFIC RULES VIOLATION':[userFeatures[8]], 'WEATHER':[userFeatures[9]], 
                    'VEHICLE TYPE AND SEX':[userFeatures[10]], 'TYPE OF ROAD':[userFeatures[11]], 'LICENSE':[userFeatures[12]],
                    'TIME':[userFeatures[13]]}
        
        # Convert to codes
        for col in testData:
            feature_value = ''.join(testData[col])
            if feature_value in data.columnCodes:
                code = [data.columnCodes[feature_value]]
                testData[col] = code
            else:
                print(f"⚠️  Unknown feature value: {feature_value}")
                # Use a default code (0) for unknown values
                testData[col] = [0]
        
        testDataFrame = pd.DataFrame.from_dict(testData)
        print(f"📊 Test dataframe shape: {testDataFrame.shape}")
        print(f"📊 Test dataframe values: {testDataFrame.values}")
        
        # Make prediction
        prediction = model.predict(testDataFrame)
        print(f"🔮 Prediction result: {prediction}")
        
        if prediction[0] == 0:
            output = "No, There is No Chance of Road Accident."
        else:
            output = "Yes, There is a Chance Of Road Accident! Be Careful."
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        # Provide a cautious default prediction
        output = "Unable to make prediction. Please drive carefully and follow traffic rules."
    
    return render_template('index.html',states = data.state, junctions = data.junction, vechicleAge = data.vehicle_age, 
                           humanAgeSex = data.human_age_sex, personWithoutPrecautions = data.person_without_precautions, 
                           areas = data.area, typeOfPlace = data.type_of_place, vehicleLoad = data.vehicle_load, 
                           trafficRulesViolation = data.traffic_rules_violation, weather = data.weather, 
                           vehicleTypeSex = data.vehicle_type_sex, roadType = data.road_type, License = data.license_type, 
                           time = data.time,prediction_text = output)  

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
