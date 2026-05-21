import streamlit as st
from live_data import get_live_data
from features import extract_features_live
from model import train_model, predict
from simulation.twin import run_simulation
import time
import csv

st.set_page_config(page_title="Bridge Monitoring System", layout="wide")

st.title("🌉 Real-Time Bridge Monitoring System")

# 🔹 Train/load model
@st.cache_resource
def get_trained_model():
    model, is_trained = train_model()
    return model, is_trained

model, is_trained = get_trained_model()

# 🔹 UI placeholders
col1, col2, col3 = st.columns(3)

weight_placeholder = col1.empty()
distance_placeholder = col2.empty()
vibration_placeholder = col3.empty()

status_placeholder = st.empty()
confidence_placeholder = st.empty()
risk_placeholder = st.empty()
twin_placeholder = st.empty()
chart_placeholder = st.empty()

# 🔹 Buffer for feature extraction
buffer = []
BUFFER_SIZE = 20

st.info("System running... collecting live data")

# 🔥 MAIN LOOP
for _ in range(100000):

    data = get_live_data()
    if data is None:
        continue

    # 🔹 Update buffer
    buffer.append(data)
    if len(buffer) > BUFFER_SIZE:
        buffer.pop(0)

    if len(buffer) < BUFFER_SIZE:
        st.warning(f"Collecting data... ({len(buffer)}/{BUFFER_SIZE})")
        time.sleep(0.5)
        continue

    # 🔹 Extract features
    features = extract_features_live(buffer)

    # 🔥 FIXED NORMALIZATION (VERY IMPORTANT)
    risk_score = (
        0.5 * (features["max_deflection"] / 50) +
        0.3 * (features["vibration_std"] / 1.5) +
        0.2 * (abs(features["deflection_rate"]) / 50)
    )

    # 🔥 SAVE DATA (MAIN FIX)
    with open("dataset.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            features["mean_deflection"],
            features["max_deflection"],
            features["vibration_std"],
            features["vibration_max"],
            features["deflection_rate"],
            "UNLABELED",
            risk_score
        ])

    # 🔹 ML Prediction
    result_ml, confidence_ml = predict(model, features, is_trained)

    # 🔥 SAFETY-FIRST DECISION LOGIC
    if result_ml is not None:
        result = result_ml
        confidence = confidence_ml

        # 🔥 OVERRIDE using physics (IMPORTANT)
        if risk_score > 0.65:
            result = "DANGER"
            confidence = max(confidence, 0.9)

        elif risk_score > 0.45 and result == "SAFE":
            result = "WARNING"

    else:
        # fallback if model not trained
        if risk_score > 0.65:
            result = "DANGER"
            confidence = 0.9
        elif risk_score > 0.45:
            result = "WARNING"
            confidence = 0.7
        else:
            result = "SAFE"
            confidence = 0.8

    # 🔹 Extract values
    weight = data["weight"]
    distance = data["distance"]
    vibration = features["vibration_std"]

    # 🔹 Display metrics
    weight_placeholder.metric("Weight (Load)", f"{weight:.2f}")
    distance_placeholder.metric("Distance (Deflection)", f"{distance:.2f} cm")
    vibration_placeholder.metric("Vibration", f"{vibration:.2f}")

    # 🔹 Risk Score display
    risk_placeholder.metric("Risk Score", f"{risk_score:.2f}")

    # 🔹 Prediction display
    if result == "SAFE":
        status_placeholder.success("✅ SAFE")
    elif result == "WARNING":
        status_placeholder.warning("⚠️ WARNING")
    else:
        status_placeholder.error("🚨 DANGER")

    confidence_placeholder.write(f"Confidence: {confidence:.2f}")

    # 🔹 Simulation (renamed correctly)
    # 🔹 Simulation (SAFE FIX - NO CRASH)

# 🔹 Simulation (FINAL FIX - SAFE)

    sim_result = None  # ✅ always define

    if data["weight"] > 0.5:
        try:
            sim_result = run_simulation(
                data["weight"],
                state=result,
                visualize=False
            )
            twin_placeholder.write(f"Estimated Structural Deflection: {sim_result:.4f}")

        except Exception:
            twin_placeholder.write("Simulation error")

    else:
        twin_placeholder.write("Simulation skipped (no load)")

    # 🔹 Chart
    chart_data = {
        "Mean Deflection": features["mean_deflection"],
        "Max Deflection": features["max_deflection"],
        "Vibration": features["vibration_std"],
        "Deflection Rate": features["deflection_rate"]
    }

    chart_placeholder.bar_chart(chart_data)

    time.sleep(1)