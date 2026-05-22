from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import os
import pickle
import json
from datetime import datetime

# Try to import SMOTE for class balancing
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

MODEL_FEATURES = [
    "mean_deflection",
    "max_deflection",
    "vibration_std",
    "vibration_max",
    "deflection_rate"
]


def train_model(csv_path="dataset.csv"):
    # 🔥 Enhanced model with class weight balancing
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight='balanced',  # ✅ Handle class imbalance
        n_jobs=-1  # Use all CPU cores
    )

    # ❌ Dataset missing
    if not os.path.exists(csv_path):
        print("⚠️ No dataset found. Model not trained.")
        return model, False

    try:
        df = pd.read_csv(csv_path)

        # ❌ Remove unlabeled data
        df = df[df["label"] != "UNLABELED"]

        # 🔥 REMOVE DUPLICATES (IMPORTANT FIX)
        df = df.drop_duplicates()

        # ❌ Not enough data
        if len(df) < 20:
            print("⚠️ Not enough labeled data. Model not trained.")
            return model, False

        # 🔥 Features & Labels
        X = df[MODEL_FEATURES]
        y = df["label"]

        # 📊 Class distribution
        print("\n📊 Class Distribution:")
        print(y.value_counts())
        print(f"   Total samples: {len(df)}")

        # 🔥 Normalize features for better training
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 🔥 Train-Test Split (for evaluation)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # ✅ Apply SMOTE for better class balance in training data
        if HAS_SMOTE:
            try:
                if len(y_train.unique()) > 1:  # Only if we have multiple classes
                    smote = SMOTE(random_state=42, k_neighbors=min(3, len(y_train) - 1))
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print("✅ SMOTE applied for class balancing")
            except Exception as e:
                print(f"⚠️ SMOTE skipped: {e}")
        else:
            print("ℹ️  SMOTE not available (install: pip install imbalanced-learn)")
            print("   Using class_weight='balanced' instead")

        # 🔥 Cross-validation with better metrics
        print("\n🔄 Running 5-Fold Cross-Validation...")
        cv_scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'f1_macro': 'f1_macro',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted'
        }

        cv_results = cross_validate(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring=cv_scoring,
            return_train_score=False
        )

        print("\n📈 Cross-Validation Results (mean ± std):")
        for metric, scores in cv_results.items():
            if metric.startswith('test_'):
                metric_name = metric.replace('test_', '')
                print(f"   {metric_name:20} → {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        # 🔥 Train final model on full training set
        model.fit(X_train, y_train)
        print("\n✅ Model trained on dataset.csv")

        # 🔥 Evaluate on test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # 📊 Comprehensive metrics
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        print("\n📊 Test Set Metrics:")
        print(f"   Accuracy:         {acc:.4f}")
        print(f"   F1 (Weighted):    {f1_weighted:.4f}")
        print(f"   F1 (Macro):       {f1_macro:.4f}")
        print(f"   Precision:        {precision:.4f}")
        print(f"   Recall:           {recall:.4f}")

        # 📊 Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📊 Confusion Matrix:")
        print(cm)

        # 📊 Classification Report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # 🔥 Feature Importance
        feature_importance = model.feature_importances_
        print("\n🎯 Feature Importance:")
        for feature, importance in zip(MODEL_FEATURES, feature_importance):
            print(f"   {feature:20} → {importance:.4f}")

        # Store scaler for prediction
        model.scaler = scaler
        
        # 💾 Save model to disk
        metrics = {
            "accuracy": float(acc),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "precision": float(precision),
            "recall": float(recall),
            "samples_trained": len(df),
            "features": MODEL_FEATURES
        }
        save_model(model, metrics)
        
        return model, True

    except Exception as e:
        print("⚠️ Training failed:", e)
        import traceback
        traceback.print_exc()
        return model, False


def predict(model, features, is_trained):
    # ❌ Model not trained
    if not is_trained:
        return None, 0.0

    try:
        X = np.array([[
            features["mean_deflection"],
            features["max_deflection"],
            features["vibration_std"],
            features["vibration_max"],
            features["deflection_rate"]
        ]])

        # ✅ Use stored scaler for consistent feature normalization
        if hasattr(model, 'scaler'):
            X = model.scaler.transform(X)

        prediction = model.predict(X)[0]

        # 🔥 Safer confidence calculation
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(X).max())
        else:
            confidence = 0.7  # fallback

        return prediction, confidence

    except Exception as e:
        print("⚠️ Prediction error:", e)
        return None, 0.0


def save_model(model, metrics, model_dir="models"):
    """Save trained model and metadata to disk"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate version timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"model_v{timestamp}"
        
        # Save model pickle
        model_path = os.path.join(model_dir, f"{version}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Model saved: {model_path}")
        
        # Save metadata
        metadata = {
            "version": version,
            "timestamp": timestamp,
            "metrics": metrics,
            "features": MODEL_FEATURES,
            "model_type": "RandomForestClassifier"
        }
        metadata_path = os.path.join(model_dir, f"{version}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved: {metadata_path}")
        
        # Update registry
        update_registry(version, metadata, model_dir)
        
    except Exception as e:
        print(f"⚠️ Failed to save model: {e}")


def load_model(model_dir="models", version=None):
    """Load trained model from disk"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        if version is None:
            # Load latest model
            registry_path = os.path.join(model_dir, "registry.json")
            if not os.path.exists(registry_path):
                print("ℹ️  No saved model found")
                return None, False
            
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            if not registry['versions']:
                print("ℹ️  No saved model found")
                return None, False
            
            # Get latest version
            latest = registry['versions'][-1]
            version = latest['version']
            print(f"✅ Loading latest model: {version}")
            print(f"   Accuracy: {latest['metrics']['accuracy']:.4f}")
        
        model_path = os.path.join(model_dir, f"{version}.pkl")
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            return None, False
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded: {version}")
        return model, True
        
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
        return None, False


def update_registry(version, metadata, model_dir="models"):
    """Update model registry with new version info"""
    try:
        registry_path = os.path.join(model_dir, "registry.json")
        
        # Load existing registry or create new
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"versions": []}
        
        # Add new version
        registry_entry = {
            "version": version,
            "timestamp": metadata["timestamp"],
            "metrics": metadata["metrics"]
        }
        registry['versions'].append(registry_entry)
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✅ Registry updated with {len(registry['versions'])} models")
        
    except Exception as e:
        print(f"⚠️ Failed to update registry: {e}")


def list_models(model_dir="models"):
    """List all saved model versions"""
    try:
        registry_path = os.path.join(model_dir, "registry.json")
        if not os.path.exists(registry_path):
            print("ℹ️  No models saved yet")
            return []
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        if not registry['versions']:
            print("ℹ️  No models saved yet")
            return []
        
        print(f"\n📦 Model Registry ({len(registry['versions'])} versions):")
        for i, entry in enumerate(registry['versions'], 1):
            acc = entry['metrics']['accuracy']
            print(f"   {i}. {entry['version']} | Accuracy: {acc:.4f} | {entry['timestamp']}")
        
        return registry['versions']
        
    except Exception as e:
        print(f"⚠️ Failed to list models: {e}")
        return []