# Test script to verify enhanced model training
from model import train_model, predict
from features import extract_features_live

print("🧪 Testing Enhanced Model Training Pipeline...\n")

# Train model with dataset
model, is_trained = train_model()

if is_trained:
    print("\n✅ Model training successful!")
    print(f"Model trained: {is_trained}")
else:
    print("⚠️ Model training skipped (not enough data or issue)")

print("\n✨ Training pipeline test complete!")
