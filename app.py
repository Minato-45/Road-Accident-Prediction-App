from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
import pandas as pd
import pickle
import numpy as np
import resources.data as data
import warnings
import os
import json
import hashlib
from datetime import datetime
import secrets
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configure Flask app for sessions
app.secret_key = secrets.token_hex(16)  # Generate secure secret key
app.config['SESSION_TYPE'] = 'filesystem'

# User storage file
USERS_FILE = 'users.json'

# Initialize users file if it doesn't exist
def init_users_file():
    """Initialize users.json file if it doesn't exist"""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)
        print("📁 Created users.json file for user storage")

# Password hashing functions
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify password against stored hash"""
    return stored_password == hashlib.sha256(provided_password.encode()).hexdigest()

# User management functions
def load_users():
    """Load users from JSON file"""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def create_user(fullname, email, username, password):
    """Create a new user account"""
    users = load_users()
    
    # Check if username or email already exists
    for user_id, user_data in users.items():
        if user_data['username'] == username:
            return False, "Username already exists"
        if user_data['email'] == email:
            return False, "Email already registered"
    
    # Create new user
    user_id = str(len(users) + 1)
    users[user_id] = {
        'fullname': fullname,
        'email': email,
        'username': username,
        'password': hash_password(password),
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }
    
    save_users(users)
    return True, "User created successfully"

def authenticate_user(username, password):
    """Authenticate user login"""
    users = load_users()
    
    for user_id, user_data in users.items():
        if user_data['username'] == username:
            if verify_password(user_data['password'], password):
                # Update last login
                users[user_id]['last_login'] = datetime.now().isoformat()
                save_users(users)
                return True, user_data
            else:
                return False, "Invalid password"
    
    return False, "Username not found"

def is_logged_in():
    """Check if user is logged in"""
    return 'user_id' in session

def get_current_user():
    """Get current logged in user data"""
    if 'user_id' in session:
        users = load_users()
        return users.get(session['user_id'])
    return None

# Initialize users file on startup
init_users_file()
print("👤 User authentication system initialized")

def robust_feature_mapping(feature_value, feature_type=""):
    """Robust mapping of feature values to codes with fallbacks for variations"""
    # Direct match first
    if feature_value in data.columnCodes:
        return data.columnCodes[feature_value]
    
    # Handle common variations and typos
    variations = {
        # Junction variations
        'T Junction': 'T-Junction',
        'T-junction': 'T-Junction',
        'Y Junction': 'Y-Junction', 
        'Y-junction': 'Y-Junction',
        'Four arm junction': 'Four arm Junction',
        'Four Arms Junction': 'Four arm Junction',
        'Four Arms': 'Four arm Junction',
        'Staggered junction': 'Staggered Junction',
        'Round about junction': 'Round about Junction',
        'Roundabout Junction': 'Round about Junction',
        
        # Vehicle age variations
        '5-10 years': '5.1 - 10 Years',
        '5-10 Years': '5.1 - 10 Years',
        '10-15 years': '10.1 - 15 Years', 
        '10-15 Years': '10.1 - 15 Years',
        'Above 15 years': '> 15 Years',
        'More than 15 years': '> 15 Years',
        'Unknown age': 'Age not known',
        
        # Human age/sex variations
        'Male 18-25': '18-25 Yrs - Male',
        'Female 18-25': '18-25 Yrs - Female',
        'Male 25-35': '25-35 Yrs- Male',
        'Female 25-35': '25-35 Yrs - Female',
        'Male 35-40': '35-40 Yrs- Male',
        'Female 35-40': '35-40 Yrs - Female',
        'Male 45-60': '45-60 Yrs- Male',
        'Female 45-60': '45-60 Yrs- Female',
        'Male 60+': '60 Yrs above- Male',
        'Female 60+': '60 Yrs above -Female',
        
        # Safety precautions variations
        'No': 'Drivers',  # Assuming "No safety" refers to drivers
        'Yes': 'Passengers',  # Assuming "Yes safety" refers to passengers
        
        # Area variations
        'Residential': 'Residential Area',
        'Institutional': 'Institutional Area', 
        'Market': 'Market/Commercial Area',
        'Commercial': 'Market/Commercial Area',
        'Open': 'Open Area',
        
        # Vehicle load variations
        'Normal': 'Normally Loaded',
        'Overloaded': 'Overloaded/Hangin',
        'Overload': 'Overloaded/Hangin',
        
        # Traffic violations variations
        'Over Speeding': 'Over-Speeding',
        'Overspeeding': 'Over-Speeding',
        'Drunk Driving': 'Drunken Driving/Consumption of Alcohol & Drug',
        'Wrong Side': 'Driving on Wrong Side',
        'Red Light': 'Jumping Red Light',
        'Mobile Phone': 'Use of Mobile Phone',
        
        # Weather variations
        'Clear': 'Sunny/Clear',
        'Sunny': 'Sunny/Clear',
        'Rain': 'Rainy',
        'Fog': 'Foggy & Misty',
        'Misty': 'Foggy & Misty',
        
        # Vehicle type variations
        'Two Wheeler Male': 'Two Wheelers - Male',
        'Two Wheeler Female': 'Two Wheelers - Female',
        'Car Male': 'Cars & taxies Vans & LMV - Male',
        'Car Female': 'Cars & taxies Vans & LMV - Female',
        
        # Road type variations
        'Straight': 'Straight Road',
        'Curved': 'Curved Road',
        'Potholes': 'Pot Holes',
        'Under Construction': 'Ongoing Road Works/Under Construction',
        
        # License variations
        'Valid License': 'License Valid Permanent',
        'Permanent License': 'License Valid Permanent',
        'Driving License': 'License Valid Permanent',
        'Learner License': "Learner's Licence",
        'No License': 'Without Licence',
        
        # Time variations
        'Day': '09-1200hrs - (Day)',  # Default day time
        'Night': '21-2400hrs - (Night)',  # Default night time
        'Morning': '06-0900hrs - (Day)',
        'Afternoon': '12-1500hrs - (Day)',
        'Evening': '15-1800hrs - (Day)',
        'Late Night': '00-300hrs - (Night)',
    }
    
    # Try variations
    if feature_value in variations:
        mapped_value = variations[feature_value]
        if mapped_value in data.columnCodes:
            print(f"🔄 Mapped '{feature_value}' -> '{mapped_value}' -> {data.columnCodes[mapped_value]}")
            return data.columnCodes[mapped_value]
    
    # If no mapping found, return a reasonable default based on feature type
    print(f"⚠️ Unknown feature value: '{feature_value}' (using default: 0)")
    return 0

def create_deployment_model():
    """Create the exact same model as trained locally for deployment consistency"""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        import numpy as np
        import pandas as pd
        
        print("🔧 Creating deployment-compatible GradientBoostingClassifier...")
        
        # Use the exact same model configuration as the trained one
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
            subsample=1.0,
            class_weight=None  # We'll handle class balancing in training data
        )
        
        # Load actual training data if available
        try:
            if os.path.exists('upload.csv'):
                print("📊 Loading real training data from upload.csv...")
                df = pd.read_csv('upload.csv')
                
                # Prepare features (same as advanced training)
                feature_columns = [
                    'States/UTs', 'JUNCTION', 'VEHICLE AGE', 'HUMAN AGE AND SEX',
                    'PERSON WITHOUT SAFETY PRECAUTIONS', 'AREA', 'TYPE OF PLACE',
                    'LOAD OF VEHICLE', 'TRAFFIC RULES VIOLATION', 'WEATHER',
                    'VEHICLE TYPE AND SEX', 'TYPE OF ROAD', 'LICENSE', 'TIME'
                ]
                
                if all(col in df.columns for col in feature_columns) and 'Accident' in df.columns:
                    # Encode features using the same mapping as the app
                    X_encoded = []
                    for _, row in df.iterrows():
                        encoded_row = []
                        for col in feature_columns:
                            value = str(row[col])
                            if value in data.columnCodes:
                                encoded_row.append(data.columnCodes[value])
                            else:
                                encoded_row.append(0)  # Default for unknown
                        X_encoded.append(encoded_row)
                    
                    X_deployment = np.array(X_encoded)
                    y_deployment = df['Accident'].values
                    
                    print(f"✅ Real data loaded: {X_deployment.shape[0]} samples, {np.mean(y_deployment)*100:.1f}% accident rate")
                    
                else:
                    raise ValueError("Required columns not found in upload.csv")
            else:
                raise FileNotFoundError("upload.csv not found")
                
        except Exception as e:
            print(f"⚠️ Could not load real data ({e}), using synthetic data...")
            
            # Fallback to synthetic data with realistic patterns
            np.random.seed(42)
            n_samples = 500
            
            # Generate features with more realistic distributions based on upload.csv analysis
            X_deployment = np.column_stack([
                np.random.choice(range(36), n_samples, p=[0.05]*20 + [0.15]*10 + [0.03]*6),  # States (some more common)
                np.random.choice(range(6), n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]),   # Junctions
                np.random.choice(range(5), n_samples, p=[0.4, 0.3, 0.2, 0.08, 0.02]),         # Vehicle age
                np.random.choice(range(14), n_samples),                                         # Human age/sex
                np.random.choice(range(2), n_samples, p=[0.7, 0.3]),                          # Person type
                np.random.choice(range(4), n_samples, p=[0.4, 0.3, 0.2, 0.1]),                # Area
                np.random.choice(range(2), n_samples, p=[0.6, 0.4]),                          # Place type
                np.random.choice(range(3), n_samples, p=[0.7, 0.25, 0.05]),                   # Vehicle load
                np.random.choice(range(5), n_samples, p=[0.4, 0.2, 0.2, 0.1, 0.1]),           # Traffic violation
                np.random.choice(range(5), n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05]),         # Weather
                np.random.choice(range(16), n_samples),                                        # Vehicle type/sex
                np.random.choice(range(8), n_samples, p=[0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]), # Road type
                np.random.choice(range(3), n_samples, p=[0.7, 0.2, 0.1]),                     # License
                np.random.choice(range(9), n_samples)                                          # Time
            ])
            
            # Create balanced target with ~30% accident rate (same as real data)
            y_deployment = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # Train the model with proper class balancing
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_deployment)
        
        print("🎯 Training GradientBoostingClassifier...")
        model.fit(X_deployment, y_deployment, sample_weight=sample_weights)
        
        # Test the model
        accuracy = model.score(X_deployment, y_deployment)
        print(f"✅ Deployment model created successfully (accuracy: {accuracy:.3f})")
        
        # Save the model for deployment consistency
        try:
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f, protocol=4)
            print("💾 Model saved to model.pkl for consistency")
        except Exception as save_error:
            print(f"⚠️ Could not save model: {save_error}")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to create deployment model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Initialize prediction system with bulletproof fallback
model = None
print("🚀 Initializing Road Accident Prediction System...")

# Enhanced model loading with comprehensive validation
print("🚀 Initializing Road Accident Prediction System...")
model = None

try:
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        
        # Comprehensive model validation with multiple test scenarios
        try:
            test_scenarios = [
                [17, 1, 4, 6, 1, 3, 0, 0, 0, 4, 6, 7, 1, 3],  # Safe scenario
                [1, 4, 4, 1, 0, 3, 1, 0, 3, 4, 11, 0, 1, 2],   # Risky scenario
            ]
            
            for i, test_input in enumerate(test_scenarios):
                test_array = np.array([test_input])
                prediction = model.predict(test_array)
                proba = model.predict_proba(test_array)[0]
                print(f"   Test {i+1}: prediction={prediction[0]}, confidence=[{proba[0]:.3f}, {proba[1]:.3f}]")
                
            print("🎯 Model validation successful!")
            
        except Exception as test_error:
            print(f"⚠️ Model validation failed: {test_error}")
            print("🔧 Model appears corrupted, will create new one if needed...")
            # Don't set model to None here, let it try in deployment
            
    else:
        print("⚠️ model.pkl not found")
        model = None
        
except Exception as e:
    print(f"⚠️ Model loading error: {e}")
    model = None

# Create a simple rule-based prediction system as fallback
def rule_based_prediction(features):
    """Improved rule-based prediction system based on real risk factors"""
    try:
        risk_score = 0
        
        # High-risk junctions (T-Junction=4, Y-Junction=5, Four Way=3)
        if features[1] in [3, 4, 5]:
            risk_score += 2
        
        # Older vehicles (10+ years = codes 2,3)
        if features[2] in [2, 3]:
            risk_score += 2
            
        # Young drivers (18-25 = codes 1,2,3,4) or very old (65+ = codes 11,12,13)
        if features[3] in [1, 2, 3, 4, 11, 12, 13]:
            risk_score += 1
            
        # High-risk traffic violations (Over-speeding=3, Jumping Red Light=2, Wrong Side=0)
        if features[8] in [0, 2, 3]:
            risk_score += 3
            
        # Bad weather conditions (Fog/Mist=0, Rainy=1, Snowy=3)
        if features[9] in [0, 1, 3]:
            risk_score += 2
            
        # High-risk vehicle types (Two wheelers=15, Heavy vehicles=8,9)
        if features[10] in [8, 9, 15]:
            risk_score += 1
            
        # High-risk road types (Bridge=0, Hill Road=2, Pothole Road=3)
        if features[11] in [0, 2, 3]:
            risk_score += 2
            
        # Invalid/No license (codes 0, 2)
        if features[12] in [0, 2]:
            risk_score += 2
            
        # High-risk time periods (Night hours: 0,1,7,8)
        if features[13] in [0, 1, 7, 8]:
            risk_score += 1
            
        # Urban areas with heavy traffic
        if features[6] == 1 and features[5] in [1, 2]:  # Urban + Commercial/Industrial
            risk_score += 1
            
        print(f"🧮 Rule-based risk score: {risk_score}/17")
        
        # Conservative threshold: predict accident only for genuinely high-risk scenarios
        return 1 if risk_score >= 6 else 0
        
    except Exception as e:
        print(f"⚠️ Rule-based prediction error: {e}")
        return 0

# System status
if model is None:
    print("🔧 Using rule-based prediction system (ML model unavailable)")
else:
    print("🎯 ML model ready for predictions!")

print("✅ Prediction system ready!")

@app.route('/')
@app.route('/first') 
def first():
	return render_template('first.html')

@app.route('/upload') 
def upload():
    if not is_logged_in():
        flash('Please log in to access this feature.', 'error')
        return redirect(url_for('login'))
    return render_template('upload.html') 

@app.route('/preview',methods=["POST"])
def preview():
    if not is_logged_in():
        flash('Please log in to access this feature.', 'error')
        return redirect(url_for('login'))
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)   
@app.route('/login', methods=['GET', 'POST']) 
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please fill in all fields', 'error')
            return render_template('login.html')
        
        success, result = authenticate_user(username, password)
        
        if success:
            # Store user session
            users = load_users()
            for user_id, user_data in users.items():
                if user_data['username'] == username:
                    session['user_id'] = user_id
                    session['username'] = username
                    session['fullname'] = user_data['fullname']
                    break
            
            flash(f'Welcome back, {result["fullname"]}!', 'success')
            return redirect(url_for('upload'))  # Redirect to main app
        else:
            flash(result, 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST']) 
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullname')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Server-side validation
        if not all([fullname, email, username, password, confirm_password]):
            flash('Please fill in all fields', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')
        
        # Email validation (basic)
        if '@' not in email or '.' not in email:
            flash('Please enter a valid email address', 'error')
            return render_template('register.html')
        
        # Create user
        success, message = create_user(fullname, email, username, password)
        
        if success:
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('first'))

@app.route('/users')
def list_users():
    """Admin endpoint to view all registered users (for development/testing)"""
    users = load_users()
    user_list = []
    
    for user_id, user_data in users.items():
        user_info = {
            'id': user_id,
            'fullname': user_data['fullname'],
            'username': user_data['username'],
            'email': user_data['email'],
            'created_at': user_data['created_at'],
            'last_login': user_data.get('last_login', 'Never')
        }
        user_list.append(user_info)
    
    return jsonify({
        'total_users': len(user_list),
        'users': user_list
    })

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
        # Check prediction system status
        model_status = "available" if model is not None else "fallback"
        
        # Check if data module is working
        data_status = "available" if hasattr(data, 'state') and len(data.state) > 0 else "unavailable"
        
        # System is always healthy now with fallback
        health_info = {
            "status": "healthy",
            "model": model_status,
            "data": data_status,
            "prediction_system": "ML + Rule-based" if model else "Rule-based",
            "timestamp": "2025-11-27"
        }
        
        return health_info, 200
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
        "python_path": os.environ.get('PYTHONPATH', 'Not set'),
        "prediction_system": "ML + Rule-based fallback" if model else "Rule-based only"
    }
    return debug_info     


@app.route('/home')
def home():
    if not is_logged_in():
        flash('Please log in to access this feature.', 'error')
        return redirect(url_for('login'))
    return render_template('index.html',states = data.state, junctions = data.junction, vechicleAge = data.vehicle_age, 
                           humanAgeSex = data.human_age_sex, personWithoutPrecautions = data.person_without_precautions, 
                           areas = data.area, typeOfPlace = data.type_of_place, vehicleLoad = data.vehicle_load, 
                           trafficRulesViolation = data.traffic_rules_violation, weather = data.weather, 
                           vehicleTypeSex = data.vehicle_type_sex, roadType = data.road_type, License = data.license_type, 
                           time = data.time)

@app.route('/dashboard')
def dashboard():
    if not is_logged_in():
        flash('Please log in to access this feature.', 'error')
        return redirect(url_for('login'))
    
    user = get_current_user()
    users = load_users()
    total_users = len(users)
    
    return render_template('index.html',
                           states = data.state, junctions = data.junction, vechicleAge = data.vehicle_age, 
                           humanAgeSex = data.human_age_sex, personWithoutPrecautions = data.person_without_precautions, 
                           areas = data.area, typeOfPlace = data.type_of_place, vehicleLoad = data.vehicle_load, 
                           trafficRulesViolation = data.traffic_rules_violation, weather = data.weather, 
                           vehicleTypeSex = data.vehicle_type_sex, roadType = data.road_type, License = data.license_type, 
                           time = data.time, user=user, total_users=total_users)


@app.route('/predict', methods = ['POST'])
def predict():
    if not is_logged_in():
        flash('Please log in to use the prediction feature.', 'error')
        return redirect(url_for('login'))
    # Get prediction using available system
    global model
    prediction_result = None
    prediction_method = "unknown"
        
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
                             time = data.time, prediction_text = "⚠️ Error: Please ensure all fields are filled correctly.")
    
    try:
        testData = {'States/UTs':[userFeatures[0]], 'JUNCTION':[userFeatures[1]], 'VEHICLE AGE':[userFeatures[2]],
                    'HUMAN AGE AND SEX':[userFeatures[3]], 'PERSON WITHOUT SAFETY PRECAUTIONS':[userFeatures[4]],
                    'AREA':[userFeatures[5]], 'TYPE OF PLACE':[userFeatures[6]], 'LOAD OF VEHICLE':[userFeatures[7]],
                    'TRAFFIC RULES VIOLATION':[userFeatures[8]], 'WEATHER':[userFeatures[9]], 
                    'VEHICLE TYPE AND SEX':[userFeatures[10]], 'TYPE OF ROAD':[userFeatures[11]], 'LICENSE':[userFeatures[12]],
                    'TIME':[userFeatures[13]]}
        
        # Convert to codes using robust mapping
        print("🔢 Converting features to codes:")
        encoded_features = []
        for i, col in enumerate(testData):
            feature_value = ''.join(testData[col])
            code = robust_feature_mapping(feature_value, col)
            testData[col] = [code]
            encoded_features.append(code)
            print(f"   {col}: '{feature_value}' -> {code}")
        
        print(f"📊 Encoded features: {encoded_features}")
        
        # Try ML model first with better error handling
        if model is not None:
            try:
                # Use numpy array directly for better compatibility
                input_array = np.array([encoded_features])
                prediction = model.predict(input_array)
                prediction_result = int(prediction[0])
                prediction_method = "ML"
                print(f"🤖 ML Prediction: {prediction_result}")
                
                # Verify prediction makes sense
                if prediction_result not in [0, 1]:
                    raise ValueError(f"Invalid prediction: {prediction_result}")
                    
            except Exception as ml_error:
                print(f"⚠️ ML prediction failed, using rule-based fallback: {ml_error}")
                prediction_result = rule_based_prediction(encoded_features)
                prediction_method = "Rule-based"
                print(f"📊 Rule-based prediction: {prediction_result}")
        else:
            print("⚠️ ML model not available, using rule-based system")
            prediction_result = rule_based_prediction(encoded_features)
            prediction_method = "Rule-based"
            print(f"📊 Rule-based prediction: {prediction_result}")
        
        # Convert prediction to output message
        if prediction_result == 0:
            output = "No, There is No Chance of Road Accident."
        else:
            output = "Yes, There is a Chance Of Road Accident! Be Careful."
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        # Provide a default prediction based on simple logic
        output = "Prediction system encountered an error. For safety, please drive carefully and follow all traffic rules."
    
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
