import pandas as pd

df = pd.read_csv("dataset.csv")

def compute_risk_score(row):
    return (
        0.5 * min(row["max_deflection"] / 50, 1) + 
        0.3 * min(row["vibration_std"] / 1.0, 1) +
        0.2 * min(abs(row["deflection_rate"]) / 30, 1)
    )

def label(row):
    score = compute_risk_score(row)

    if score > 0.7:
        return "DANGER"
    elif score > 0.4:
        return "WARNING"
    else:
        return "SAFE"

df["risk_score"] = df.apply(compute_risk_score, axis=1)
df["label"] = df.apply(label, axis=1)

df.to_csv("dataset.csv", index=False)

print("✅ Dataset labeled using improved risk score")