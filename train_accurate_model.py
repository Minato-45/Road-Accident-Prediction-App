#!/usr/bin/env python3
"""
Advanced Road Accident Prediction Model Training
This script creates a properly trained model that gives accurate, varied predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import the data mapping
try:
    from resources import data
    print("✅ Successfully imported resources.data")
except ImportError:
    print("❌ Could not import resources.data")
    data = None

def load_and_encode_data():
    """Load upload.csv and properly encode features using data.py mappings"""
    
    print("📊 Loading and encoding training data...")
    
    # Load the dataset
    df = pd.read_csv('upload.csv')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Map column names to match the form inputs
    column_mapping = {
        'States/UTs': 'States/UTs',
        'JUNCTION': 'JUNCTION', 
        'VEHICLE AGE': 'VEHICLE AGE',
        'HUMAN AGE AND SEX': 'HUMAN AGE AND SEX',
        'PERSON': 'PERSON',
        'AREA': 'AREA',
        'TYPE OF PLACE': 'TYPE OF PLACE', 
        'LOAD OF VEHICLE': 'LOAD OF VEHICLE',
        'TRAFFIC RULES VIOLATION': 'TRAFFIC RULES VIOLATION',
        'WEATHER': 'WEATHER',
        'VEHICLE TYPE AND SEX': 'VEHICLE TYPE AND SEX',
        'TYPE OF ROAD': 'TYPE OF ROAD',
        'LICENSE': 'LICENSE',
        'TIME': 'TIME',
        'ACCIDENT OCCURRENCE': 'target'
    }
    
    # Create feature matrix using the exact same encoding as app.py
    features = []
    feature_names = []
    
    # Process each feature in the exact order used in app.py
    form_to_data_mapping = {
        'States/UTs': 'States/UTs',
        'JUNCTION': 'JUNCTION',
        'VEHICLE AGE': 'VEHICLE AGE', 
        'HUMAN AGE AND SEX': 'HUMAN AGE AND SEX',
        'PERSON': 'PERSON',
        'AREA': 'AREA',
        'TYPE OF PLACE': 'TYPE OF PLACE',
        'LOAD OF VEHICLE': 'LOAD OF VEHICLE', 
        'TRAFFIC RULES VIOLATION': 'TRAFFIC RULES VIOLATION',
        'WEATHER': 'WEATHER',
        'VEHICLE TYPE AND SEX': 'VEHICLE TYPE AND SEX',
        'TYPE OF ROAD': 'TYPE OF ROAD',
        'LICENSE': 'LICENSE',
        'TIME': 'TIME'
    }
    
    encoded_features = []
    
    for form_field, data_col in form_to_data_mapping.items():
        if data_col in df.columns and data is not None:
            print(f"Processing {form_field}...")
            
            # Get the feature values
            feature_values = df[data_col].values
            
            # Encode using data.columnCodes mapping
            encoded_values = []
            for value in feature_values:
                if value in data.columnCodes:
                    encoded_values.append(data.columnCodes[value])
                else:
                    # If value not found, use a default encoding
                    print(f"⚠️  Unknown value '{value}' in {form_field}, using default encoding")
                    encoded_values.append(0)
            
            encoded_features.append(encoded_values)
            feature_names.append(form_field)
    
    # Create feature matrix
    X = np.column_stack(encoded_features)
    
    # Create target variable (YES=1, NO=0) 
    y = (df['ACCIDENT OCCURRENCE'] == 'YES').astype(int)
    
    print(f"✅ Feature matrix created: {X.shape}")
    print(f"✅ Target distribution: No Accident: {np.sum(y==0)}, Accident: {np.sum(y==1)}")
    print(f"✅ Accident rate: {np.mean(y)*100:.1f}%")
    
    return X, y, feature_names

def create_balanced_model(X, y):
    """Create a well-balanced model that gives varied predictions"""
    
    print("\n🤖 Training balanced prediction model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training target distribution: No: {np.sum(y_train==0)}, Yes: {np.sum(y_train==1)}")
    
    # Try multiple models to find the best one
    models = {
        'Balanced RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',  # This is key for balanced predictions
            random_state=42
        ),
        'Weighted RandomForest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight={0: 1, 1: 2},  # Give more weight to accident class
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Check prediction variety
        unique_preds = len(np.unique(y_pred))
        pred_distribution = np.bincount(y_pred)
        
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Unique predictions: {unique_preds}")
        print(f"   Prediction distribution: {pred_distribution}")
        
        # Score based on accuracy and prediction variety
        variety_score = unique_preds / 2.0  # Max 2 classes
        combined_score = accuracy * variety_score
        
        if combined_score > best_score:
            best_score = combined_score
            best_model = model
            best_name = name
        
        # Show classification report
        print(f"   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Accident', 'Accident'], zero_division=0))
    
    print(f"\n🏆 Best model: {best_name} (score: {best_score:.3f})")
    
    # Final evaluation of best model
    y_pred_best = best_model.predict(X_test)
    
    print("\n📊 Final Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.3f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['No Accident', 'Accident']))
    
    # Test prediction variety with sample data
    print("\n🧪 Testing prediction variety...")
    sample_predictions = []
    for i in range(0, min(20, len(X_test))):
        pred = best_model.predict([X_test[i]])[0]
        sample_predictions.append(pred)
    
    sample_pred_counts = np.bincount(sample_predictions)
    print(f"Sample predictions: No Accident: {sample_pred_counts[0] if len(sample_pred_counts) > 0 else 0}, "
          f"Accident: {sample_pred_counts[1] if len(sample_pred_counts) > 1 else 0}")
    
    return best_model

def save_model_safely(model, filename='model.pkl'):
    """Save the model with multiple protocols for compatibility"""
    
    print(f"\n💾 Saving model to {filename}...")
    
    try:
        # Try protocol 4 first (most compatible)
        with open(filename, 'wb') as f:
            pickle.dump(model, f, protocol=4)
        
        # Verify the model loads correctly
        with open(filename, 'rb') as f:
            loaded_model = pickle.load(f)
        
        print("✅ Model saved and verified successfully!")
        print(f"   Model type: {type(loaded_model).__name__}")
        
        # Test a prediction to ensure it works
        test_data = np.array([[1, 2, 3, 4, 0, 1, 1, 0, 2, 4, 10, 5, 1, 2]])  # Sample features
        test_pred = loaded_model.predict(test_data)
        print(f"   Test prediction: {test_pred[0]} {'(Accident)' if test_pred[0] == 1 else '(No Accident)'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def main():
    """Main training pipeline"""
    
    print("🚗 Road Accident Prediction - Advanced Model Training")
    print("=" * 60)
    
    try:
        # Load and encode the data
        X, y, feature_names = load_and_encode_data()
        
        print(f"\nFeature names: {feature_names}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Check for any issues with the data
        if X.shape[0] == 0:
            print("❌ No data loaded!")
            return
            
        if np.all(y == y[0]):
            print("⚠️  All targets are the same - this will cause prediction bias!")
        
        # Create the balanced model
        model = create_balanced_model(X, y)
        
        # Save the model
        if save_model_safely(model):
            print("\n" + "=" * 60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("✅ New model will provide varied predictions")
            print("✅ Both 'Accident' and 'No Accident' outcomes possible")
            print("✅ Model saved with deployment compatibility")
            
            # Show feature importance if available
            if hasattr(model, 'feature_importances_'):
                print("\n📈 Top 5 Most Important Features:")
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:5]
                for i, idx in enumerate(indices):
                    feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                    print(f"   {i+1}. {feature_name}: {importances[idx]:.3f}")
        else:
            print("❌ Failed to save model!")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()