import numpy as np

def extract_features(data):
    features = {}

    features["mean_deflection"] = float(data["deflection"].mean())
    features["max_deflection"] = float(data["deflection"].max())
    features["vibration_std"] = float(data["vibration"].std())
    features["vibration_max"] = float(data["vibration"].max())
    features["deflection_rate"] = float(data["deflection"].diff().mean())

    return features

def extract_features_live(data):
    # 🔹 Raw values
    weight = data["weight"]
    distance = data["distance"]
    ax = data["ax"]
    ay = data["ay"]
    az = data["az"]

    # 🔹 Clean noise
    if abs(weight) < 5:
        weight = 0

    if distance <= 0 or distance > 400:
        distance = 400

    # 🔹 Vibration magnitude
    vibration = abs(ax) + abs(ay) + abs(az)

    # 🔹 Normalize vibration (important)
    vibration = vibration / 1000.0

    return {
        "mean_deflection": distance,
        "max_deflection": distance,
        "vibration_std": vibration,
        "vibration_max": vibration,
        "deflection_rate": weight
    }