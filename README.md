# Road Accident Prediction System

A machine learning-powered web application that predicts road accident probability using data mining techniques and comprehensive traffic analysis.

[![GitHub](https://img.shields.io/badge/Source%20Code-GitHub-blue)](https://github.com/Minato-45/Road-Accident-Prediction-App)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-red)](https://flask.palletsprojects.com/)

## Project Overview

A comprehensive web-based prediction system that analyzes 14 critical traffic and environmental factors to predict road accident probability with high accuracy. Features complete user authentication, responsive design, and data visualization capabilities.

## Key Features

### Core Functionality
- **AI-Powered Predictions**: Machine Learning model with high accuracy
- **User Authentication**: Secure registration, login, and session management
- **Responsive Design**: Fully responsive interface for all devices
- **Real-Time Analysis**: Instant prediction results
- **Data Visualization**: Interactive charts and performance analytics
- **Dataset Management**: Upload and analyze custom datasets
- **Secure Access Control**: Protected routes with user session management

### Authentication System
- **User Registration**: Secure account creation with validation
- **Login/Logout**: Username/password authentication with session tracking
- **Route Protection**: Prediction features require authentication
- **Data Security**: SHA-256 password hashing and secure storage
- **User Management**: Registration tracking and account management

### Prediction Parameters
The system analyzes **14 critical risk factors**:

| Parameter | Description | Impact |
|-----------|-------------|--------|
| **State/UT** | 36 Indian states and territories | Regional risk patterns |
| **Junction Type** | T-Junction, Y-Junction, Four-arm, etc. | Traffic complexity |
| **Vehicle Age** | Age categories of vehicles | Vehicle reliability |
| **Demographics** | Age and gender combinations | Driver risk profiles |
| **Safety Measures** | Driver vs. Passenger precautions | Safety compliance |
| **Area Type** | Residential, Commercial, Institutional | Traffic density |
| **Location** | Urban vs. Rural classification | Infrastructure quality |
| **Vehicle Load** | Normal, Overloaded, Other | Vehicle stability |
| **Traffic Violations** | Over-speeding, Wrong side, Red light | Rule compliance |
| **Weather** | Clear, Rainy, Foggy, Hail | Environmental conditions |
| **Vehicle Category** | Two-wheeler, Car, Bus, Truck + Gender | Vehicle-specific risks |
| **Road Type** | Straight, Curved, Bridge, Pothole | Infrastructure safety |
| **License Status** | Valid, Learner's, Without license | Driver qualification |
| **Time Period** | AM/PM time slots with Day/Night | Temporal risk factors |

## Technology Stack

### Backend
- **Python 3.11+**: Core programming language
- **Flask 2.3.2**: Web framework with session management
- **Authentication**: SHA-256 password hashing and secure sessions
- **JSON Storage**: Local user data storage
- **scikit-learn 1.3.0**: Machine learning library
- **pandas 2.0.2**: Data manipulation and analysis
- **NumPy 1.24.3**: Numerical computing
- **Gunicorn 20.1.0**: Production WSGI server

### Frontend
- **HTML5/CSS3**: Modern semantic structure and styling
- **Responsive CSS**: Mobile-first responsive design
- **JavaScript/jQuery**: Interactive functionality
- **Bootstrap**: Responsive grid system
- **Custom Themes**: Professional UI design

### Machine Learning
- **Gradient Boosting Classifier**: Advanced ensemble learning
- **Feature Engineering**: Categorical encoding with robust mapping
- **Class Balancing**: Handles imbalanced datasets
- **Dual System**: ML model with rule-based fallback
- **Cross-Validation**: Rigorous testing methodology

## Machine Learning Model

### Model Performance
- **Overall Accuracy**: 93%+
- **Precision**: High accuracy for both classes
- **Recall**: Balanced prediction across outcomes
- **Class Distribution**: Handles imbalanced datasets effectively

### Training Data
- **Dataset Size**: Real-world accident records
- **Geographic Coverage**: Multiple Indian states and territories
- **Data Quality**: Validated and cleaned records
- **Feature Engineering**: 14 categorical variables with proper encoding

### Model Architecture
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=8,
    random_state=42,
    subsample=1.0,
    class_weight='balanced'
)
```

## Application Structure

```
Road Accident Prediction System
├── Homepage (/)                  → Landing page and project overview
├── User Registration (/register) → Account creation with validation
├── User Login (/login)           → Secure authentication system
├── Logout (/logout)              → Session termination
├── Data Upload (/upload)         → CSV dataset management (Protected)
├── Data Preview (/preview)       → Dataset visualization (Protected)
├── Prediction Interface (/dashboard) → Main ML prediction tool (Protected)
├── Performance Analytics (/performance) → Model metrics
├── Data Charts (/chart)          → Interactive visualizations
├── User Management (/users)      → API endpoint for user data
├── System Health (/healthz)      → Application status monitoring
└── Debug Information (/debug)    → Development information
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- pip package manager
- Git version control

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Minato-45/Road-Accident-Prediction-App.git
cd Road-Accident-Prediction-App

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Open in browser
# Navigate to: http://localhost:5000
```

### Development Mode
```bash
# For development with auto-reload
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn --bind 0.0.0.0:5000 app:app

# Using Docker
docker build -t road-accident-app .
docker run -p 5000:5000 road-accident-app
```

## Usage

### Authentication Flow
1. **Registration**: Create account with full name, email, username, and password
2. **Login**: Authenticate with username and password
3. **Session Management**: Persistent login across browser sessions
4. **Protected Access**: Prediction features require authentication
5. **Logout**: Secure session termination

### Prediction Workflow
1. **Login Required**: Authenticate to access prediction features
2. **Form Interface**: Select values from 14 dropdown parameters
3. **Processing**: Advanced model analysis with fallback system
4. **Results**: Color-coded predictions with risk assessment

### Example Scenarios

#### High-Risk Scenario
```yaml
State: "Uttar Pradesh"
Junction: "Four arm Junction"
Vehicle Age: "> 15 Years"
Demographics: "18-25 Yrs - Male"
Traffic Violation: "Over-Speeding"
Weather: "Foggy & Misty"
Time: "Night"
Result: ⚠️ "Accident risk detected"
```

#### Low-Risk Scenario
```yaml
State: "Kerala"
Junction: "Others"
Vehicle Age: "Less than 5 years"
Demographics: "35-40 Yrs - Female"
Traffic Violation: "None"
Weather: "Sunny/Clear"
Time: "Day"
Result: ✅ "No accident risk"
```

## 🔧 Configuration

### Environment Variables
```bash
# Application Configuration
PORT=5000                       # Server port (default: 5000)
FLASK_ENV=production           # Environment mode
FLASK_DEBUG=False              # Debug mode (development only)
SECRET_KEY=auto-generated      # Session security (auto-generated)
```

### System Health Monitoring
```bash
# Health Check Endpoint
GET /healthz

# Example Response
{
  "status": "healthy",
  "model": "available",
  "data": "available", 
  "prediction_system": "ML + Rule-based",
  "timestamp": "2024-11-28"
}

# Debug Information
GET /debug

# Example Response
{
  "model_loaded": true,
  "model_type": "GradientBoostingClassifier",
  "data_states_count": 36,
  "session_info": {
    "user_logged_in": true,
    "session_keys": ["user_id", "username", "fullname"]
  }
}
```

## API Documentation

### Authentication Endpoints
```bash
# User Registration
POST /register
{
  "fullname": "John Doe",
  "email": "john@example.com",
  "username": "johndoe", 
  "password": "securepassword"
}

# User Login
POST /login
{
  "username": "johndoe",
  "password": "securepassword"
}

# Logout
GET /logout
```

### Prediction Endpoints
```bash
# Get Prediction Form
GET /dashboard
Authorization: Session-based (login required)

# Submit Prediction
POST /predict
Authorization: Session-based (login required)
Content-Type: application/x-www-form-urlencoded
```

### System Endpoints
```bash
# Health Check
GET /healthz

# Debug Information
GET /debug

# User List
GET /users
```

## Contributing

```bash
# Fork the repository
# Create a feature branch
git checkout -b feature/your-feature

# Make changes and test
python app.py

# Commit changes
git commit -m "Add your feature"

# Push to branch
git push origin feature/your-feature

# Open a Pull Request
```

## License

This project is developed for educational and research purposes in road safety and accident prevention.

## Contact

- **Developer**: Minato-45
- **Repository**: [Road-Accident-Prediction-App](https://github.com/Minato-45/Road-Accident-Prediction-App)
- **Issues**: [GitHub Issues](https://github.com/Minato-45/Road-Accident-Prediction-App/issues)

## Support

If this project helped you, please consider:
- ⭐ Star this repository on GitHub
- 🍴 Fork and contribute to the codebase
- 🐛 Report issues to help improve the system
- 💡 Suggest features for future development

---

**Built for road safety through data science and machine learning technology**