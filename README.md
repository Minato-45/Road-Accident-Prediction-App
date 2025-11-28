# 🚗 Road Accident Prediction System

<div align="center">

**AI-powered accident prediction with 93.1% accuracy to prevent road incidents**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Try_Now-brightgreen?style=for-the-badge&logo=render)](https://road-accident-prediction-app.onrender.com)
[![GitHub Stars](https://img.shields.io/github/stars/Minato-45/Road-Accident-Prediction-App?style=for-the-badge&logo=github&color=yellow)](https://github.com/Minato-45/Road-Accident-Prediction-App)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-red?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)

**🎯 93.1% Accuracy** | **📱 Mobile Ready** | **🔐 Secure Authentication** | **⚡ Real-time Prediction**

</div>

## ✨ Features

- 🎯 **Real-time accident prediction** with 93.1% ML accuracy
- 🔐 **Complete authentication system** with secure session management  
- 📱 **Mobile-responsive design** for all devices
- 📊 **Interactive data visualization** with Google Charts
- 🛡️ **Dual prediction system** with ML model + rule-based fallback
- ⚡ **Lightning-fast performance** with sub-second responses

## 🚀 Quick Start

### 🌐 Try Live Demo (No Setup Required)
1. Visit: [road-accident-prediction-app.onrender.com](https://road-accident-prediction-app.onrender.com)
2. Register your account
3. Login to access prediction features
4. Select parameters and get instant predictions

### 💻 Run Locally
```bash
git clone https://github.com/Minato-45/Road-Accident-Prediction-App.git
cd Road-Accident-Prediction-App
pip install -r requirements.txt
python app.py
# Open: http://localhost:5000
```

## 🧠 AI Model Performance

| Metric | Score | Description |
|:------:|:-----:|:-----------:|
| 🎯 **Accuracy** | **93.1%** | Overall prediction accuracy |
| 📊 **Precision** | **94%** / **90%** | No Accident / Accident classes |
| 🔄 **Recall** | **Balanced** | Cross-class performance |
| 🛡️ **Fallback** | **85%+** | Rule-based backup system |

**Model Details:**
- **Algorithm:** Gradient Boosting Classifier with 100 estimators
- **Dataset:** 576 real-world Indian road accident records
- **Features:** 14 critical risk factors (weather, road type, demographics, etc.)
- **Validation:** Train/validation/test splits with cross-validation

## 🎯 Prediction Factors

The system analyzes **14 critical risk factors:**

| Category | Parameters | Examples |
|----------|------------|----------|
| 📍 **Location** | State/UT, Junction Type, Area Type | "Karnataka", "T-Junction", "Urban" |
| 🚗 **Vehicle** | Age, Type, Load Status | "<5 Years", "Car", "Normal Load" |
| 👤 **Human** | Demographics, Safety, License | "Male 25-35", "Precautions", "Valid" |
| 🌍 **Environment** | Weather, Time, Road Type | "Clear", "Day", "Straight Road" |

<details>
<summary>📋 <strong>Complete Parameter List</strong> (Click to expand)</summary>

1. **State/UT** - 36 Indian states and territories
2. **Junction Type** - T-Junction, Y-Junction, Four-arm, etc.
3. **Vehicle Age** - <5, 5-10, 10-15, >15 years  
4. **Demographics** - Age and gender combinations
5. **Safety Precautions** - Driver/Passenger compliance
6. **Area Type** - Residential, Commercial, Rural
7. **Location Type** - Urban vs Rural classification
8. **Vehicle Load** - Normal, Overloaded, Other
9. **Traffic Violations** - Speed, Wrong side, Red light
10. **Weather Conditions** - Clear, Rain, Fog, Hail
11. **Vehicle Category** - Two-wheeler, Car, Bus, Truck
12. **Road Type** - Straight, Curved, Bridge, Potholes
13. **License Status** - Valid, Learner's, Without
14. **Time Period** - AM/PM with Day/Night classification

</details>

## 🚀 Technology Stack

**Backend:**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)

**Frontend:**
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E)
![Bootstrap](https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white)

**Deployment:**
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)

## 📊 Example Predictions

### 🔴 High-Risk Scenario
**Input:** Uttar Pradesh, Four-arm Junction, >15 Years Old Vehicle, Male 18-25, Night Time, Foggy Weather, No Precautions  
**Result:** ⚠️ **HIGH RISK (87% probability)** - Avoid travel, use public transport

### 🟢 Low-Risk Scenario  
**Input:** Kerala, Straight Road, <5 Years Old Vehicle, Female 35-45, Day Time, Clear Weather, Safety Precautions  
**Result:** ✅ **LOW RISK (8% probability)** - Safe to travel with caution

## 📱 Application Features

- 🔐 **User Authentication** - Secure registration and login system
- 📊 **Interactive Charts** - Age distribution and state-wise analytics  
- 📱 **Mobile Responsive** - Works perfectly on all devices
- 📈 **Performance Dashboard** - Model metrics and confusion matrix
- 💾 **Data Management** - Upload and analyze custom datasets

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

**Areas for contribution:**
- 🎨 UI/UX improvements
- 🤖 ML model enhancements
- 📱 Mobile optimization  
- 🔐 Security improvements
- 📊 Data visualizations

## 📞 Contact & Support

- **🌐 Live Demo:** [road-accident-prediction-app.onrender.com](https://road-accident-prediction-app.onrender.com)
- **📱 GitHub:** [Minato-45/Road-Accident-Prediction-App](https://github.com/Minato-45/Road-Accident-Prediction-App)
- **🐛 Issues:** [Report bugs](https://github.com/Minato-45/Road-Accident-Prediction-App/issues)
- **💡 Discussions:** [Feature requests](https://github.com/Minato-45/Road-Accident-Prediction-App/discussions)

## ⭐ Support This Project

If you find this project helpful, please consider:

[![Star Repository](https://img.shields.io/badge/⭐_Star_this_Repository-yellow?style=for-the-badge)](https://github.com/Minato-45/Road-Accident-Prediction-App)
[![Fork Repository](https://img.shields.io/badge/🍴_Fork_Repository-blue?style=for-the-badge)](https://github.com/Minato-45/Road-Accident-Prediction-App/fork)

---

<div align="center">

**Built with ❤️ for road safety through AI and modern web technology**

</div>
