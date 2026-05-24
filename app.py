import streamlit as st
from live_data import get_live_data
from features import extract_features_live
from model import load_model, predict, list_models
from simulation.twin import run_simulation
import time
import csv

# ============================================
# DATA COLLECTION CONTROL
# ============================================
COLLECT_DATA = False  # Disabled: read-only monitoring mode keeps dataset.csv locked
# Set to True only for data collection phase
# ============================================

st.set_page_config(page_title="Bridge Monitoring System", layout="wide")

st.title("Real-Time Bridge Monitoring System")

# Load latest trained model from disk (NO CACHING)
@st.cache_resource
def get_latest_model():
    """Load latest model from disk"""
    model, is_trained = load_model()
    if not is_trained:
        st.error("No trained model found! Run: python train_improved_model.py")
        st.stop()
    return model, is_trained

# Display model info
st.sidebar.header("Model Information")
with st.sidebar:
    model, is_trained = get_latest_model()
    st.success("Model Loaded")
    
    # Status
    if COLLECT_DATA:
        st.warning("Data Collection: ON")
    else:
        st.info("Data Collection: OFF (read-only monitoring)")
    
    st.info("To use new model:\n1. Train: `python train_improved_model.py`\n2. Press 'C' to clear cache\n3. Refresh page")
    
    # Show model registry
    with st.expander("Model History"):
        try:
            list_models()
        except:
            pass
    
    st.divider()
    st.caption("Training data is locked at 2,826 samples\nAll data is fully labeled and validated")

# UI placeholders
col1, col2, col3 = st.columns(3)

weight_placeholder = col1.empty()
distance_placeholder = col2.empty()
vibration_placeholder = col3.empty()

status_placeholder = st.empty()
confidence_placeholder = st.empty()
risk_placeholder = st.empty()
twin_placeholder = st.empty()
chart_placeholder = st.empty()

# Buffer for feature extraction
buffer = []
BUFFER_SIZE = 20

st.info("System running in READ-ONLY mode (no data collection)")

# MAIN LOOP
for _ in range(100000):

    data = get_live_data()
    if data is None:
        time.sleep(0.02)
        continue

    # Update buffer
    buffer.append(data)
    if len(buffer) > BUFFER_SIZE:
        buffer.pop(0)

    if len(buffer) < BUFFER_SIZE:
        st.warning(f"Collecting data... ({len(buffer)}/{BUFFER_SIZE})")
        time.sleep(0.5)
        continue

    # Extract features
    features = extract_features_live(buffer)

    # Fixed risk normalization
    risk_score = (
        0.5 * (features["max_deflection"] / 50) +
        0.3 * (features["vibration_std"] / 1.5) +
        0.2 * (abs(features["deflection_rate"]) / 50)
    )

    # Auto-save is disabled unless data collection is explicitly enabled.
    if COLLECT_DATA:
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

    # ML Prediction
    result_ml, confidence_ml = predict(model, features, is_trained)

    # Safety-first decision logic
    if result_ml is not None:
        result = result_ml
        confidence = confidence_ml

        # Override using physics when risk score is high.
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

    # Extract values
    weight = data["weight"]
    distance = data["distance"]
    vibration = features["vibration_std"]

    # Display metrics
    weight_placeholder.metric("Weight (Load)", f"{weight:.2f}")
    distance_placeholder.metric("Distance (Deflection)", f"{distance:.2f} cm")
    vibration_placeholder.metric("Vibration", f"{vibration:.2f}")

    # Risk Score display
    risk_placeholder.metric("Risk Score", f"{risk_score:.2f}")

    # Prediction display
    if result == "SAFE":
        status_placeholder.success("SAFE")
    elif result == "WARNING":
        status_placeholder.warning("WARNING")
    else:
        status_placeholder.error("DANGER")

    confidence_placeholder.write(f"Confidence: {confidence:.2f}")

    # Simulation
    # Keep simulation failures from stopping monitoring.

# Simulation block

    sim_result = None

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

    # Chart
    chart_data = {
        "Mean Deflection": features["mean_deflection"],
        "Max Deflection": features["max_deflection"],
        "Vibration": features["vibration_std"],
        "Deflection Rate": features["deflection_rate"]
    }

    chart_placeholder.bar_chart(chart_data)

    time.sleep(1)
