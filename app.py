import streamlit as st
from live_data import get_live_data
from features import extract_features_live
from model import train_model, predict
from simulation.twin import run_simulation
import time

st.set_page_config(page_title="Bridge Monitoring Digital Twin", layout="wide")

st.title("🌉 Real-Time Bridge Monitoring System")

# 🔹 Train model once
@st.cache_resource
def get_trained_model():
    model = train_model()

    # Dummy training (temporary but required)
    X = [
        [2,3,0.5,0.6,10],   # SAFE
        [5,6,0.9,1.0,50],   # WARNING
        [8,10,1.5,1.8,100]  # DANGER
    ]
    y = ["SAFE", "WARNING", "DANGER"]

    model.fit(X, y)
    return model

model = get_trained_model()

# 🔹 Live placeholders
col1, col2, col3 = st.columns(3)

weight_placeholder = col1.empty()
distance_placeholder = col2.empty()
vibration_placeholder = col3.empty()

status_placeholder = st.empty()
confidence_placeholder = st.empty()
twin_placeholder = st.empty()
chart_placeholder = st.empty()

# 🔥 LIVE LOOP
while True:
    data = get_live_data()
    features = extract_features_live(data)

    result, confidence = predict(model, features)

    # 🔹 Extract values
    weight = data["weight"]
    distance = data["distance"]
    vibration = features["vibration_std"]

    # 🔹 Display metrics
    weight_placeholder.metric("Weight (Load)", f"{weight:.2f}")
    distance_placeholder.metric("Distance (Deflection)", f"{distance:.2f} cm")
    vibration_placeholder.metric("Vibration", f"{vibration:.2f}")

    # 🔹 Prediction display
    if result == "SAFE":
        status_placeholder.success("✅ SAFE")
    elif result == "WARNING":
        status_placeholder.warning("⚠️ WARNING")
    else:
        status_placeholder.error("🚨 DANGER")

    confidence_placeholder.write(f"Confidence: {confidence:.2f}")

    # 🔹 Digital Twin
    sim_result = run_simulation(
        features["mean_deflection"],
        state=result,
        visualize=False
    )

    twin_placeholder.write(f"Digital Twin Deflection (uy): {sim_result:.4f}")

    # 🔹 Feature visualization
    chart_data = {
        "Mean Deflection": features["mean_deflection"],
        "Max Deflection": features["max_deflection"],
        "Vibration": features["vibration_std"],
        "Deflection Rate": features["deflection_rate"]
    }

    chart_placeholder.bar_chart(chart_data)

    time.sleep(1)