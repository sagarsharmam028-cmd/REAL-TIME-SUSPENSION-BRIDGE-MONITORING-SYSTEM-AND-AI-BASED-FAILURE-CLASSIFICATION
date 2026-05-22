# Direct training script - doesn't require serial connection
from model import train_model, list_models

print("🚀 Training and saving first production model...\n")

# Train model (will automatically save to disk)
model, is_trained = train_model()

if is_trained:
    print("\n✅ Model successfully trained and saved!")
    list_models()
else:
    print("\n⚠️ Model training failed or not enough data")
