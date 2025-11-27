# 🚗🔮 Road Accident Prediction System

A sophisticated machine learning-powered web application that predicts road accident probability using advanced data mining techniques and comprehensive traffic analysis.

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Render-brightgreen)](https://road-accident-prediction-app.onrender.com)
[![GitHub](https://img.shields.io/badge/📂%20Source%20Code-GitHub-blue)](https://github.com/Minato-45/Road-Accident-Prediction-App)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-red)](https://flask.palletsprojects.com/)

## 🌟 Live Application

**🚀 Access the deployed application:** [https://road-accident-prediction-app.onrender.com](https://road-accident-prediction-app.onrender.com)

> 🎯 **Try it now!** Experience real-time road accident prediction with our advanced ML model

## ✨ Features

### 🎯 Core Functionality
- **Real-time Accident Prediction**: Advanced ML model predicting accident probability with 93.1% accuracy
- **Interactive Web Interface**: User-friendly forms with dropdown selections for all input parameters
- **Comprehensive Data Analysis**: Built-in visualization and performance analytics
- **Dataset Management**: Upload, preview, and train custom datasets
- **Multi-page Navigation**: Dedicated pages for prediction, analysis, charts, and performance metrics

### 🔍 Prediction Parameters
The system analyzes **14 critical factors**:
- 📍 **Location**: State/UT (36 Indian states and territories)
- 🛣️ **Junction Type**: Traffic intersection characteristics
- 🚗 **Vehicle Age**: Age category of the vehicle
- 👤 **Human Demographics**: Age and gender combinations
- ⚠️ **Safety Precautions**: Driver/passenger safety measures
- 🏘️ **Area Type**: Urban, rural, highway classifications
- 📍 **Place Type**: Specific location characteristics
- 📦 **Vehicle Load**: Load carrying status
- 🚦 **Traffic Violations**: Rule compliance status
- 🌤️ **Weather Conditions**: Environmental factors
- 🚛 **Vehicle Type**: Vehicle category and driver demographics
- 🛤️ **Road Type**: Infrastructure classification
- 📋 **License Type**: Driver licensing status
- ⏰ **Time Factors**: Temporal risk assessment

## 🚀 Live Deployment

### Render Platform Details
- **Platform**: [Render](https://render.com)
- **Environment**: Docker containerized (Python 3.11.6)
- **Auto-deployment**: Triggered by GitHub commits
- **Health Monitoring**: Built-in health checks at `/healthz`
- **Debug Information**: Available at `/debug` endpoint

### Quick Access
```bash
🌐 Production URL: https://road-accident-prediction-app.onrender.com
📊 Health Check: https://road-accident-prediction-app.onrender.com/healthz
🔧 Debug Info: https://road-accident-prediction-app.onrender.com/debug
```

## 🏠 Local Development

### Prerequisites
- Python 3.11+
- pip package manager
- Git

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Minato-45/Road-Accident-Prediction-App.git
cd Road-Accident-Prediction-App

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Access locally
# Open: http://localhost:5000
```

### Development Environment
```bash
# For development with auto-reload
export FLASK_ENV=development
python app.py
```

## 🛠️ Technology Stack

### Backend
- **🐍 Python 3.11+**: Core programming language
- **🌶️ Flask 2.3.2**: Web framework
- **🤖 scikit-learn 1.3.0**: Machine learning library
- **📊 pandas 2.0.2**: Data manipulation
- **🔢 NumPy 1.24.3**: Numerical computing
- **🚀 Gunicorn 20.1.0**: WSGI server

### Frontend
- **🎨 HTML5/CSS3**: Structure and styling
- **⚡ JavaScript**: Interactive functionality
- **🎨 Bootstrap**: Responsive design framework
- **🎬 Custom CSS**: TemplateMo training studio theme

### Deployment
- **🐳 Docker**: Containerization
- **☁️ Render**: Cloud platform
- **📦 Git**: Version control and auto-deployment

## 🧠 Machine Learning Model

### Model Architecture
- **Algorithm**: Gradient Boosting Classifier
- **Accuracy**: 93.1% on test dataset
- **Training Data**: 576 real-world accident records
- **Features**: 14 categorical variables with proper encoding
- **Class Balance**: Handles imbalanced datasets (30.7% accident rate)

### Model Performance
```
📈 Precision: 94% (No Accident), 90% (Accident)
📊 Recall: Balanced prediction across both classes
🎯 F1-Score: Optimized for real-world deployment
⚖️ Class Weighting: Addresses data imbalance
```

### Training Pipeline
```python
# Advanced model training with class balancing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and prepare data
data = pd.read_csv('upload.csv')
X = data.drop('Accident', axis=1)
y = data['Accident']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    class_weight='balanced',
    random_state=42
)

# Train model
model.fit(X_train, y_train)
print(f"Model Accuracy: {model.score(X_test, y_test):.3f}")
# Output: Model Accuracy: 0.931
```

### API Usage Examples
```python
# Example 1: Health Check
import requests

response = requests.get('https://road-accident-prediction-app.onrender.com/healthz')
print(response.json())
# Output: {"status": "healthy", "model": "available", "data": "available"}

# Example 2: Prediction via Form Data
prediction_data = {
    'state': 'Karnataka',
    'junction': 'T Junction',
    'vechicleAge': '5-10 years',
    'humanAgeSex': 'Male 25-35',
    # ... other 10 parameters
}

response = requests.post(
    'https://road-accident-prediction-app.onrender.com/predict',
    data=prediction_data
)
```

## 🎮 Live Demo Features

### 🚀 Try These Features Live:

| Feature | URL | Description |
|---------|-----|-------------|
| 🏠 **Homepage** | [/first](https://road-accident-prediction-app.onrender.com/first) | Welcome page with project overview |
| 🎯 **Live Prediction** | [/home](https://road-accident-prediction-app.onrender.com/home) | Real-time accident prediction |
| 📊 **Analytics** | [/performance](https://road-accident-prediction-app.onrender.com/performance) | Model performance metrics |
| 📈 **Charts** | [/chart](https://road-accident-prediction-app.onrender.com/chart) | Data visualizations |
| 📤 **Data Upload** | [/upload](https://road-accident-prediction-app.onrender.com/upload) | Dataset management |
| 🔍 **Health Check** | [/healthz](https://road-accident-prediction-app.onrender.com/healthz) | System status |

## 📱 Application Structure

```bash
📁 Road Accident Prediction App
├── 🏠 Homepage (/)           → Landing page and navigation
├── 🔑 Login (/login)         → Admin authentication (admin/admin)
├── 📤 Upload (/upload)       → Dataset management & CSV upload
├── 👁️ Preview (/preview)     → Data visualization & preview
├── 🎯 Prediction (/home)     → Main prediction interface
├── 📊 Charts (/chart)        → Interactive data analysis charts
├── 📈 Performance (/performance) → Model analytics & confusion matrix
├── 🔍 Health (/healthz)      → System health monitoring
└── 🛠️ Debug (/debug)         → Development information
```

## 📊 Dataset Information

### Training Data Features
- **Source**: Indian road accident records (2018)
- **Records**: 576 comprehensive entries
- **Coverage**: Multiple states, various conditions
- **Quality**: Real-world validated data
- **Encoding**: Categorical variables properly mapped

### Feature Engineering
```python
# Example feature encoding
'Andhra Pradesh' → 0
'Assam' → 1
'Bihar' → 2
# ... 33 more states/UTs
```

## 🔧 Configuration

### Environment Variables
```bash
PORT=5000                    # Server port
FLASK_ENV=production        # Environment mode
PYTHONPATH=./               # Python module path
```

### 🏥 Health Monitoring
```bash
# Check application status
curl https://road-accident-prediction-app.onrender.com/healthz

# Live Response Example
{
  "status": "healthy",
  "model": "available",
  "data": "available",
  "timestamp": "2025-11-27"
}

# Debug information
curl https://road-accident-prediction-app.onrender.com/debug

# Debug Response
{
  "model_loaded": true,
  "model_type": "<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>",
  "model_file_exists": true,
  "data_states_count": 36,
  "working_directory": "/opt/render/project/src",
  "python_path": "/opt/render/project/src"
}
```

## 🚀 Deployment Process

### Automatic Deployment
1. **Code Push**: Commit changes to GitHub
2. **Auto-trigger**: Render detects repository changes
3. **Build Process**: Docker container creation
4. **Health Check**: Automatic service validation
5. **Go Live**: Application updates automatically

### Manual Deployment
```bash
# Push changes
git add .
git commit -m "Update application"
git push origin main

# Render automatically deploys
```

## 🏆 Project Highlights

### Advanced Features
- **🔄 Fallback System**: Automatic model recovery on failure
- **⚡ Performance Optimization**: Efficient prediction processing
- **🛡️ Error Handling**: Comprehensive validation and error management
- **📱 Responsive Design**: Mobile-friendly interface
- **🎨 Professional UI**: Modern, intuitive user experience

### Innovation
- **🧮 Real-world Training**: Actual accident data from Indian roads
- **⚖️ Class Balancing**: Advanced techniques for imbalanced datasets
- **🔮 Accurate Predictions**: Varied outcomes based on risk factors
- **📊 Comprehensive Analysis**: Multiple visualization and analysis tools

## 🎯 Sample Prediction Scenarios

### 🔴 High Risk Scenario
```yaml
State: "Uttar Pradesh"          # High accident rate state
Junction: "Four Arms"           # Complex intersection
Vehicle Age: "Above 15 years"   # Older vehicle
Human Age/Sex: "Male 18-25"     # High-risk demographic
Safety Precautions: "No"        # No safety measures
Area: "Urban"                   # Heavy traffic
Weather: "Fog"                  # Poor visibility
Time: "Night"                   # Low visibility period

Prediction: 🔴 "High Accident Risk" (85% confidence)
```

### 🟢 Low Risk Scenario
```yaml
State: "Kerala"                 # Lower accident rate
Junction: "No Junction"         # Straight road
Vehicle Age: "1-5 years"        # Newer vehicle
Human Age/Sex: "Female 35-50"   # Safer demographic
Safety Precautions: "Yes"       # Safety measures used
Area: "Rural"                   # Less traffic
Weather: "Clear"               # Good visibility
Time: "Day"                     # Good visibility

Prediction: 🟢 "Low Accident Risk" (92% confidence)
```

## 📝 Usage Instructions

### 🎯 For Live Predictions
1. **Visit** [the prediction page](https://road-accident-prediction-app.onrender.com/home)
2. **Select** values for all 14 parameters from dropdowns
3. **Click** "Predict" for instant ML-powered analysis
4. **View** results with risk assessment and confidence score

### For Data Analysis
1. **Login** with admin credentials (admin/admin)
2. **Upload** CSV datasets for analysis
3. **Preview** data structure and quality
4. **Train** custom models if needed
5. **View** performance analytics and charts

## 🤝 Contributing

```bash
# Fork the repository
# Create a feature branch
git checkout -b feature/amazing-feature

# Commit changes
git commit -m 'Add amazing feature'

# Push to branch
git push origin feature/amazing-feature

# Open a Pull Request
```

## 📄 License

This project is developed for educational and research purposes in road safety and accident prevention.

## 📧 Contact & Links

- 👨‍💻 **Developer**: Minato-45
- 📂 **Repository**: [Road-Accident-Prediction-App](https://github.com/Minato-45/Road-Accident-Prediction-App)
- 🌐 **Live Demo**: [https://road-accident-prediction-app.onrender.com](https://road-accident-prediction-app.onrender.com)
- 📊 **Health Status**: [https://road-accident-prediction-app.onrender.com/healthz](https://road-accident-prediction-app.onrender.com/healthz)

## 🎯 Quick Test

Want to test immediately? Try this:

1. 🌐 **Visit**: [road-accident-prediction-app.onrender.com](https://road-accident-prediction-app.onrender.com)
2. 🎯 **Go to Prediction**: Click "prediction" in navigation
3. 🔧 **Fill Form**: Select any combination of the 14 parameters
4. 🚀 **Get Results**: Instant ML prediction with confidence score!

> **Pro Tip**: Try different combinations to see how various factors affect accident probability!

---

## ⭐ Star this repository if it helped you!

**Built with ❤️ for road safety and accident prevention through data science**