#!/usr/bin/env python3
"""
Create a compatible model for the current sklearn version
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def create_compatible_model():
    """Create a new model compatible with current sklearn version"""
    print("Creating compatible model...")
    
    # Create synthetic training data based on the original features
    np.random.seed(42)
    
    # Generate realistic synthetic data for road accident prediction
    n_samples = 1000
    
    # Features: 14 columns as per the original model
    # [State, Junction, Vehicle_Age, Human_Age_Sex, Safety_Precautions, Area, 
    #  Place_Type, Vehicle_Load, Traffic_Rules, Weather, Vehicle_Type_Sex, 
    #  Road_Type, License, Time]
    
    X = np.random.randint(0, 36, size=(n_samples, 14))  # Random values in realistic ranges
    
    # Synthetic target: accidents more likely with certain combinations
    # Simple risk-based logic
    risk_score = (
        X[:, 2] * 0.1 +      # Vehicle age
        X[:, 4] * 0.2 +      # Safety precautions 
        X[:, 8] * 0.3 +      # Traffic rules violation
        X[:, 9] * 0.15 +     # Weather
        X[:, 11] * 0.1 +     # Road type
        X[:, 12] * 0.15      # License
    )
    
    # Convert to binary classification (accident or no accident)
    y = (risk_score > np.median(risk_score)).astype(int)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model (same architecture as original)
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Test accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save the new compatible model
    with open('model_new.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Compatible model saved as 'model_new.pkl'")
    return model

if __name__ == "__main__":
    create_compatible_model()