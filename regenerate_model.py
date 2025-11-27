"""
Regenerate model with exact same configuration for deployment consistency
This ensures both local and deployed versions use identical models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import resources.data as data
import warnings
warnings.filterwarnings('ignore')

def regenerate_consistent_model():
    """Create the exact model configuration that should be used everywhere"""
    print("🚀 Regenerating consistent model for deployment...")
    
    try:
        # Load training data
        print("📊 Loading training data...")
        df = pd.read_csv('upload.csv')
        print(f"   Loaded {len(df)} samples")
        
        # Prepare features - use actual column names from upload.csv
        feature_columns = [
            'States/UTs', 'JUNCTION', 'VEHICLE AGE', 'HUMAN AGE AND SEX',
            'PERSON', 'AREA', 'TYPE OF PLACE',
            'LOAD OF VEHICLE', 'TRAFFIC RULES VIOLATION', 'WEATHER',
            'VEHICLE TYPE AND SEX', 'TYPE OF ROAD', 'LICENSE', 'TIME'
        ]
        
        # Check if columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print(f"   Available columns: {list(df.columns)}")
            raise ValueError(f"Required columns not found: {missing_cols}")
        
        # Check target column - it's 'ACCIDENT OCCURRENCE' not 'Accident'
        target_column = 'ACCIDENT OCCURRENCE'
        if target_column not in df.columns:
            print(f"❌ Target column '{target_column}' not found")
            print(f"   Available columns: {list(df.columns)}")
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Encode features
        print("🔢 Encoding features...")
        X_encoded = []
        encoding_stats = {}
        
        for _, row in df.iterrows():
            encoded_row = []
            for col in feature_columns:
                value = str(row[col])
                if value in data.columnCodes:
                    code = data.columnCodes[value]
                    encoded_row.append(code)
                    # Track encoding stats
                    if col not in encoding_stats:
                        encoding_stats[col] = {}
                    if value not in encoding_stats[col]:
                        encoding_stats[col][value] = 0
                    encoding_stats[col][value] += 1
                else:
                    print(f"   ⚠️ Unknown value '{value}' in column '{col}' -> using 0")
                    encoded_row.append(0)
            X_encoded.append(encoded_row)
        
        X = np.array(X_encoded)
        
        # Convert target column to binary (YES/NO -> 1/0)
        print("🎯 Processing target column...")
        y_raw = df[target_column].values
        y = np.array([1 if str(val).upper() == 'YES' else 0 for val in y_raw])
        
        print(f"   Target conversion: {list(np.unique(y_raw))} -> {list(np.unique(y))}")
        
        print(f"   Features shape: {X.shape}")
        print(f"   Target distribution: {np.bincount(y)} ({np.mean(y)*100:.1f}% accidents)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create model with EXACT same configuration
        print("🤖 Creating GradientBoostingClassifier...")
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
            subsample=1.0,
            criterion='friedman_mse',
            init=None,
            loss='deviance',
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_iter_no_change=None,
            tol=0.0001,
            validation_fraction=0.1,
            verbose=0,
            warm_start=False
        )
        
        # Train with balanced weights
        print("🎯 Training model with balanced weights...")
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate
        print("📈 Evaluating model...")
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        
        print(f"   Training accuracy: {train_score:.3f}")
        print(f"   Test accuracy: {test_score:.3f}")
        print(f"   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Accident', 'Accident']))
        
        # Test prediction variety
        print("🧪 Testing prediction variety...")
        sample_predictions = []
        for i in range(min(10, len(X_test))):
            pred = model.predict([X_test[i]])[0]
            prob = model.predict_proba([X_test[i]])[0]
            sample_predictions.append(pred)
            print(f"   Sample {i+1}: Prediction={pred}, Probabilities={prob}")
        
        unique_predictions = len(set(sample_predictions))
        print(f"   Prediction variety: {unique_predictions}/2 classes represented")
        
        if unique_predictions < 2:
            print("   ⚠️ WARNING: Model shows limited prediction variety!")
        
        # Save model with maximum compatibility
        print("💾 Saving model...")
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=4)  # Use protocol 4 for broad compatibility
        
        print("✅ Consistent model generated and saved!")
        
        # Verify saved model
        print("🔍 Verifying saved model...")
        with open('model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test the loaded model
        test_input = X_test[0:1]  # Single sample
        original_pred = model.predict(test_input)[0]
        loaded_pred = loaded_model.predict(test_input)[0]
        
        if original_pred == loaded_pred:
            print("✅ Model verification successful - predictions match!")
        else:
            print(f"❌ Model verification failed - predictions don't match: {original_pred} vs {loaded_pred}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error regenerating model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = regenerate_consistent_model()
    if success:
        print("\n🎉 Model regeneration completed successfully!")
        print("   The model.pkl file now contains a consistent, high-quality model")
        print("   that should work identically in both local and deployed environments.")
    else:
        print("\n💥 Model regeneration failed!")
        print("   Please check the error messages above and try again.")