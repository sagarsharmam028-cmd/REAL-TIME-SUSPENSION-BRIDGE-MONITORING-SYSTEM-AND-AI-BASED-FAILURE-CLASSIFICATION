from live_data import get_live_data
from features import extract_features_live
from model import train_model, predict
from simulation.twin import run_simulation
import csv
import os

# 🔹 Create model
model = train_model()

# 🔹 Temporary training (initial boot)
X = [
    [2,3,0.5,0.6,10],   # SAFE
    [5,6,0.9,1.0,50],   # WARNING
    [8,10,1.5,1.8,100]  # DANGER
]
y = ["SAFE", "WARNING", "DANGER"]

model.fit(X, y)

print("✅ Model ready. Starting live system...\n")

# 🔹 Create dataset file if not exists
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
    data = get_live_data()
    features = extract_features_live(data)

    result, confidence = predict(model, features)

    print("\nLIVE DATA:", data)
    print("FEATURES:", features)
    print("PREDICTION:", result, "| Confidence:", round(confidence, 2))

    # 🔹 Digital twin
    sim_result = run_simulation(
        features["mean_deflection"],
        state=result,
        visualize=False
    )

    print("Digital Twin Deflection:", sim_result)

    # 🔴 OPTIONAL: Save data for training
    save = input("Save this data? (y/n): ")

    if save.lower() == "y":
        label = input("Enter correct label (SAFE/WARNING/DANGER): ")

        with open(file_name, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                features["mean_deflection"],
                features["max_deflection"],
                features["vibration_std"],
                features["vibration_max"],
                features["deflection_rate"],
                label
            ])

        print("✅ Data saved!")