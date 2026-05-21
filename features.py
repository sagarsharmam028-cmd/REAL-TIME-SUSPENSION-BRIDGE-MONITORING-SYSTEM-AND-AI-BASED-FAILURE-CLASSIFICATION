import numpy as np

def extract_features_live(buffer):
    # 🔹 Extract values from buffer
    distances = [d["distance"] for d in buffer]
    weights = [d["weight"] for d in buffer]

    # 🔹 Proper vibration magnitude (physics-based)
    vibrations = [
        np.sqrt(d["ax"]**2 + d["ay"]**2 + d["az"]**2)
        for d in buffer
    ]

    # 🔹 Features (REAL now)
    mean_deflection = np.mean(distances)
    max_deflection = np.max(distances)

    vibration_std = np.std(vibrations)
    vibration_max = np.max(vibrations)

    # 🔹 Rate of change (trend)
    deflection_rate = distances[-1] - distances[0]

    return {
        "mean_deflection": round(mean_deflection, 3),
        "max_deflection": round(max_deflection, 3),
        "vibration_std": round(vibration_std, 3),
        "vibration_max": round(vibration_max, 3),
        "deflection_rate": round(deflection_rate, 3)
    }