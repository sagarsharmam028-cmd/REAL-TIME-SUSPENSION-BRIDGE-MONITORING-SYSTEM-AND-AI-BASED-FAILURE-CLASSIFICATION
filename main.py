import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from live_data import get_live_data
from features import extract_features_live
from model import train_model, predict, load_model, list_models
from simulation.twin import run_simulation
import csv
import os
import time

# 🔹 Buffer for time-series analysis
buffer = []
BUFFER_SIZE = 20

# 🔹 Try to load existing model first, otherwise train new one
print("🔍 Checking for saved models...")
model, is_trained = load_model()

if not is_trained:
    print("📚 No saved model found. Training new model...\n")
    model, is_trained = train_model()

# 🔹 List all available models
if is_trained:
    list_models()

print("✅ System started. Live monitoring running...\n")

# 🔹 Dataset file setup
file_name = "dataset.csv"

if not os.path.exists(file_name):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mean_deflection",
            "max_deflection",
            "vibration_std",
            "vibration_max",
            "deflection_rate",
            "label"
        ])

# 🔥 LIVE LOOP
while True:
    try:
        # ✅ Step 1: Get live data
        data = get_live_data()
        if data is None:
            time.sleep(0.02)
            continue

        # ✅ Step 2: Update buffer
        buffer.append(data)
        if len(buffer) > BUFFER_SIZE:
            buffer.pop(0)

        # ❗ Wait until buffer fills
        if len(buffer) < BUFFER_SIZE:
            print(f"⏳ Collecting data... ({len(buffer)}/{BUFFER_SIZE})")
            continue

        # ✅ Step 3: Extract features
        features = extract_features_live(buffer)

        # ✅ Step 4: Try ML prediction
        result_ml, confidence_ml = predict(model, features, is_trained)

        # ✅ Step 5: Rule-based fallback / override
        # ✅ NEW (calibrated for your sensor)
        if result_ml is not None:
            result = result_ml
            confidence = confidence_ml
        else:
            if features["max_deflection"] > 60:
                result = "DANGER"
                confidence = 1.0
            elif features["max_deflection"] > 40:
                result = "WARNING"
                confidence = 0.7
            else:
                result = "SAFE"
                confidence = 0.9

        # ✅ Display
        print("\nLIVE DATA:", data)
        print("FEATURES:", features)
        print("PREDICTION:", result, "| Confidence:", round(confidence, 2))

        # ✅ Digital Twin
        sim_result = run_simulation(
            features["mean_deflection"],
            state=result,
            visualize=False
        )

        print("Expected Deflection:", sim_result)

        # ✅ Auto-save (for future ML training)
        with open(file_name, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                features["mean_deflection"],
                features["max_deflection"],
                features["vibration_std"],
                features["vibration_max"],
                features["deflection_rate"],
                "UNLABELED"
            ])

    except Exception as e:
        print("⚠️ Error in loop:", e)