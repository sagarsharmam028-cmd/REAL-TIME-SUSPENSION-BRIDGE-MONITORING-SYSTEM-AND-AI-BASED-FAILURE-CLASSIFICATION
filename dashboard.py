#!/usr/bin/env python3
"""
Bridge Monitoring System - Premium Streamlit Dashboard
Features:
- Asynchronous threaded serial reader integration
- Advanced signal processing features (FFT, dynamic regression deflection slope)
- Physics-informed digital twin visualizer (SVG-based bending bridge deck)
- Interactive premium Plotly chart overlays
- Modern glassmorphic dark-theme aesthetics with custom CSS
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime
import plotly.graph_objects as go

# Import our robust hardware pipeline, features extraction, and model modules
import live_data
from live_data import get_live_data, serial_port_name, send_serial_command
from features import extract_features_live
from model import load_model, predict, list_models, MODEL_FEATURES
from simulation.twin import run_simulation

# ============================================
# PERSISTENT STORAGE & CALIBRATION CACHING
# ============================================
@st.cache_resource
def get_persistent_history():
    return []

@st.cache_resource
def get_calibration_state():
    return {
        "baseline_distance": 71.5,
        "is_calibrated": False
    }

plotly_history = get_persistent_history()
calib_state = get_calibration_state()

# ============================================
# PAGE SETUP & STYLING
# ============================================
st.set_page_config(
    page_title="Bridge Structural Digital Twin Dashboard",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Global Styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphic Cards targeting Streamlit's native bordered containers */
    div[data-testid="stVerticalBlockBordered"] {
        background: rgba(25, 31, 46, 0.7) !important;
        border: 1.5px dashed rgba(0, 210, 255, 0.25) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.55) !important;
        backdrop-filter: blur(12px) !important;
        margin-bottom: 20px !important;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    }
    div[data-testid="stVerticalBlockBordered"]:hover {
        border-color: rgba(0, 210, 255, 0.6) !important;
        box-shadow: 0 16px 48px 0 rgba(0, 210, 255, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Header Gradient styling */
    .title-gradient {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #00d2ff 0%, #0066ff 50%, #9b51e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .subtitle-text {
        color: #90A4AE;
        font-size: 1.1rem;
        margin-bottom: 25px;
        font-weight: 300;
    }
    
    /* Metrics adjustments */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #00d2ff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        color: #B0BEC5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Section
st.markdown('<div class="title-gradient">🌉 Bridge Structural Health Digital Twin</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Low-cost IoT & Physics-ML Cyber-Physical Bridge Monitoring & Failure Classification</div>', unsafe_allow_html=True)

# ============================================
# SIDEBAR - MODEL REGISTRY & HARDWARE
# ============================================
st.sidebar.markdown("### 📊 System Status & Control")

# Interactive switch to toggle monitoring
run_live = st.sidebar.checkbox("🟢 Enable Active Monitoring", value=True, help="Toggle serial polling and prediction updates.")

# Load model info
@st.cache_resource
def get_model_info():
    model, is_trained = load_model()
    if not is_trained:
        return None, False
    return model, True

model, is_trained = get_model_info()

if not is_trained:
    st.sidebar.error("❌ No trained model found!")
    st.sidebar.info("**To train a model:**\n1. Label data: `python labeling.py`\n2. Train: `python train_improved_model.py`\n3. Refresh this page")
    st.stop()

# Model metrics loading
try:
    registry_path = "models/registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        if registry['versions']:
            latest = registry['versions'][-1]
            st.sidebar.success("✅ ML Model Loaded")
            st.sidebar.metric("Active ML Model", latest['version'].replace('model_v', 'v'))
            st.sidebar.caption(f"**Trained on:** {latest['metrics']['samples_trained']} samples")
            
            # Simple Expanders for metrics & history
            with st.sidebar.expander("📈 Model Metrics"):
                st.write(f"**Model Type:** {latest['metrics'].get('model_type', 'RandomForest')}")
                st.write(f"**Accuracy:** {latest['metrics']['accuracy']:.2%}")
                st.write(f"**F1 Score:** {latest['metrics']['f1_weighted']:.4f}")
                st.write(f"**Precision:** {latest['metrics']['precision']:.4f}")
                st.write(f"**Recall:** {latest['metrics']['recall']:.4f}")
except Exception as e:
    st.sidebar.warning(f"Metadata load error: {e}")

# Hardware Status Sidebar HUD
st.sidebar.divider()
st.sidebar.markdown("### 🔌 Telemetry Controls")

# Manual selector for simulated data override
data_source = st.sidebar.selectbox(
    "Data Source Mode",
    ["🔌 Real Hardware (Auto-detect)", "💻 Simulated Mode"],
    index=1 if live_data.FORCE_SIMULATION else 0,
    help="Select whether to read from physical USB serial hardware or generate realistic simulated physics events."
)

# Detect dynamic mode switch to reset baseline calibration appropriately
new_force_sim = (data_source == "💻 Simulated Mode")
if new_force_sim != live_data.FORCE_SIMULATION:
    live_data.FORCE_SIMULATION = new_force_sim
    if new_force_sim:
        calib_state["baseline_distance"] = 3.0  # Simulated zero-load height
        calib_state["is_calibrated"] = True
        st.toast("💻 Switched to Simulated Mode. Baseline set to 3.0 cm.", icon="💻")
    else:
        calib_state["baseline_distance"] = 71.5  # Physical default height
        calib_state["is_calibrated"] = False  # Triggers automatic tare calibration on first read
        st.toast("🔌 Switched to Real Hardware. Baseline will auto-calibrate on next read.", icon="🔌")

if live_data.USE_REAL:
    st.sidebar.success(f"Connected: COM / Serial Port Active")
else:
    st.sidebar.info("Simulated Mode Active")

# Add manual baseline calibration button
if st.sidebar.button("🔌 Calibrate Sensor Baseline (Tare)", help="Click when the bridge is completely empty to set the baseline distance."):
    if "latest_distance" in st.session_state:
        calib_state["baseline_distance"] = st.session_state.latest_distance
        calib_state["is_calibrated"] = True
        st.sidebar.success(f"Baseline calibrated to {calib_state['baseline_distance']:.2f} cm")
        st.toast("✅ Sensor calibrated successfully!", icon="🔌")
    else:
        st.sidebar.warning("Wait for first sensor reading before calibrating.")

# ============================================
# PHYSICS-INFORMED BRIDGE VISUAL ENGINE (SVG)
# ============================================
def draw_bridge_3d_html(deflection, load, status, elements_data_json):
    """Generates a premium, interactive 3D WebGL structural representation of the bridge deck bending under load using Three.js"""
    # Select color hex and text indicators based on warning/danger classification
    if status == "SAFE":
        color = "#00E676"      # Bright neon green
        glow_hex = "0x00E676"
    elif status == "WARNING":
        color = "#FFD600"      # Bright warning gold
        glow_hex = "0xFFD600"
    else:
        color = "#FF1744"      # Glowing crimson red
        glow_hex = "0xFF1744"

    # HTML page containing WebGL 3D Canvas, OrbitControls, and Three.js structural assembly
    # Double curly braces are used to escape them from the Python f-string parser
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; background-color: #101423; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
            #canvas {{ width: 100vw; height: 100vh; }}
            #hud {{
                position: absolute; top: 10px; left: 10px;
                color: #ECEFF1; font-family: 'JetBrains Mono', monospace; font-size: 11px;
                background: rgba(16, 20, 35, 0.85); padding: 10px 14px; border-radius: 10px;
                border: 1px solid rgba(255,255,255,0.08); pointer-events: none;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
                z-index: 10;
            }}
            .hud-line {{ margin-bottom: 3px; }}
            .hud-title {{ font-weight: bold; font-size: 10px; letter-spacing: 1px; color: #90A4AE; margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <div id="hud">
            <div class="hud-title">3D DIGITAL TWIN TELEMETRY</div>
            <div class="hud-line" style="font-weight: bold; color: {color};">STATE: {status}</div>
            <div class="hud-line">DECK BEND: {deflection:.2f} cm</div>
            <div class="hud-line">LOAD MASS: {load:.2f} kg</div>
            <div style="font-size: 8px; margin-top: 6px; color: #607D8B; font-weight: bold;">* Left-click + drag to ROTATE in 3D</div>
            <div style="font-size: 8px; color: #607D8B; font-weight: bold;">* Scroll to ZOOM</div>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script>
            // 🔹 Initialize Three.js Scene, Camera, and WebGL Renderer
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x101423);
            scene.fog = new THREE.FogExp2(0x101423, 0.04);

            const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 100);
            camera.position.set(11, 4.5, 9.5);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);

            // 🔹 Setup Interactive Orbit Controls (Zoom, Pan, Orbit)
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxPolarAngle = Math.PI / 2 - 0.03;
            controls.minDistance = 4;
            controls.maxDistance = 22;
            controls.target.set(0, 0.2, 0);

            const stateHex = {glow_hex};

            // 🔹 Add Ambient & Directional Lighting
            const ambient = new THREE.AmbientLight(0xffffff, 0.35);
            scene.add(ambient);

            const sunLight = new THREE.DirectionalLight(0xffffff, 0.9);
            sunLight.position.set(15, 30, 10);
            scene.add(sunLight);

            // Add dynamic glowing warning light at center span
            const warningLight = new THREE.PointLight(stateHex, 2.2, 10);
            warningLight.position.set(0, 0.5, 0);
            scene.add(warningLight);

            // 🔹 Water/River Ground Plane
            const waterGeo = new THREE.PlaneGeometry(40, 40);
            const waterMat = new THREE.MeshStandardMaterial({{
                color: 0x091122,
                roughness: 0.12,
                metalness: 0.85
            }});
            const water = new THREE.Mesh(waterGeo, waterMat);
            water.rotation.x = -Math.PI / 2;
            water.position.y = -2.2;
            scene.add(water);

            // 🔹 Structural Materials
            const steelMat = new THREE.MeshStandardMaterial({{ color: 0x607D8B, metalness: 0.8, roughness: 0.2 }});
            const concreteMat = new THREE.MeshStandardMaterial({{ color: 0x37474F, metalness: 0.4, roughness: 0.5 }});

            // Abutments
            const lAbutment = new THREE.Mesh(new THREE.BoxGeometry(1.4, 2, 2.2), concreteMat); lAbutment.position.set(-6.8, -1.2, 0); scene.add(lAbutment);
            const rAbutment = new THREE.Mesh(new THREE.BoxGeometry(1.4, 2, 2.2), concreteMat); rAbutment.position.set(6.8, -1.2, 0); scene.add(rAbutment);

            // 🔹 Dynamic elements parsing and structural rendering
            const elements = {elements_data_json};
            
            // Helper function to map FEM coordinates to Three.js coordinates
            function mapFEMToThree(coord) {{
                const x = coord[0] - 5.0;
                const y = coord[2] - 1.0;
                const z = coord[1] - 1.0;
                return new THREE.Vector3(x, y, z);
            }}

            // Helper function to create a cylinder mesh between two points
            function createCylinderBetweenPoints(pointA, pointB, radius, material) {{
                const direction = new THREE.Vector3().subVectors(pointB, pointA);
                const height = direction.length();
                if (height < 0.001) return null;
                
                const geometry = new THREE.CylinderGeometry(radius, radius, height, 6);
                const mesh = new THREE.Mesh(geometry, material);
                
                const midpoint = new THREE.Vector3().addVectors(pointA, pointB).multiplyScalar(0.5);
                mesh.position.copy(midpoint);
                
                direction.normalize();
                const alignAxis = new THREE.Vector3(0, 1, 0);
                const quaternion = new THREE.Quaternion().setFromUnitVectors(alignAxis, direction);
                mesh.setRotationFromQuaternion(quaternion);
                
                return mesh;
            }}

            elements.forEach(elem => {{
                // Determine radius based on element type
                let radius = 0.02;
                if (elem.type === 'tower') radius = 0.065;
                else if (elem.type === 'cable') radius = 0.045;
                else if (elem.type === 'hanger') radius = 0.012;
                else if (elem.type === 'deck') radius = 0.035;
                else if (elem.type === 'brace') radius = 0.018;

                // Map displaced coordinates
                const pointA = mapFEMToThree(elem.node_i_displaced);
                const pointB = mapFEMToThree(elem.node_j_displaced);

                // Color calculation based on structural force (Tension = Red, Compression = Yellow/Orange)
                const force = elem.force;
                const maxForce = 15000.0; // Reference force limit for full glow scaling (15kN)
                
                let baseColor = new THREE.Color(0x3A506B); // Stable steel blue
                if (elem.type === 'tower') baseColor = new THREE.Color(0x455A64);
                else if (elem.type === 'cable') baseColor = new THREE.Color(0x00E676);
                else if (elem.type === 'hanger') baseColor = new THREE.Color(0x90A4AE);
                else if (elem.type === 'deck') baseColor = new THREE.Color(0x00B0FF);

                let matColor = baseColor.clone();
                let emissiveIntensity = 0.05;

                if (force > 5.0) {{
                    // Tension -> glow red
                    const t = Math.min(1.0, force / maxForce);
                    matColor.lerp(new THREE.Color(0xFF1744), t);
                    emissiveIntensity = 0.1 + t * 0.9;
                }} else if (force < -5.0) {{
                    // Compression -> glow yellow-orange
                    const t = Math.min(1.0, Math.abs(force) / maxForce);
                    matColor.lerp(new THREE.Color(0xFFD600), t);
                    emissiveIntensity = 0.1 + t * 0.9;
                }}

                const elemMat = new THREE.MeshStandardMaterial({{
                    color: matColor,
                    roughness: 0.25,
                    metalness: 0.75,
                    emissive: matColor,
                    emissiveIntensity: emissiveIntensity
                }});

                const cylinder = createCylinderBetweenPoints(pointA, pointB, radius, elemMat);
                if (cylinder) {{
                    scene.add(cylinder);
                }}
            }});

            // 🔹 Dynamically scale vehicle sphere load at deck center
            if ({load} > 0.1) {{
                const vehicle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.34, 16, 16),
                    new THREE.MeshStandardMaterial({{ color: 0xFF9100, metalness: 0.9, roughness: 0.1 }})
                );
                // Position vehicle centered on displaced deck height
                vehicle.position.set(0, 0.1 - {deflection} * 0.1 + 0.34, 0);
                scene.add(vehicle);
            }}

            // 🔹 Real-time WebGL Render Loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();

            // Handle browser resize fluidly
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html_content



# ============================================
# MAIN DASHBOARD TABS
# ============================================
tab1, tab2, tab3 = st.tabs(["🔴 Real-Time Digital Twin", "📈 ML Model Registry", "📊 Raw Signal Spectrum"])

with tab1:
    # 2x2 grid layout inside Live tab
    hud_col, stats_col = st.columns([1.1, 0.9])
    
    with hud_col:
        with st.container(border=True):
            st.subheader("🌉 Live Dynamic Twin Representation")
            twin_svg_placeholder = st.empty()
        
    with stats_col:
        with st.container(border=True):
            st.subheader("💡 Predictive Telemetry")
            
            # Grid of metrics
            met1, met2 = st.columns(2)
            weight_metric = met1.empty()
            deflection_metric = met2.empty()
            
            met3, met4 = st.columns(2)
            vibration_metric = met3.empty()
            confidence_metric = met4.empty()
            
            status_label_placeholder = st.empty()

    # Plotly dynamic history graph
    with st.container(border=True):
        st.subheader("📊 Dynamic Telemetry Waveform")
        plotly_chart_placeholder = st.empty()

with tab3:
    st.subheader("📊 Advanced Signal Feature Spectrum")
    st.write("Real-time Fast Fourier Transform (FFT) dynamic frequency analysis of structural cable resonances.")
    
    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        dom_freq_metric = st.empty()
    with spec_col2:
        spec_energy_metric = st.empty()
        
    spectrum_chart_placeholder = st.empty()

with tab2:
    st.header("📦 Saved Production Models Analytics")
    st.write("Historical versions tracked and registered in `/models/registry.json`")
    
    try:
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            if registry['versions']:
                metrics_data = []
                for entry in registry['versions']:
                    metrics_data.append({
                        'Version': entry['version'].replace('model_v', 'v'),
                        'Accuracy': f"{entry['metrics']['accuracy']:.2%}",
                        'Weighted F1': f"{entry['metrics']['f1_weighted']:.4f}",
                        'Precision': f"{entry['metrics']['precision']:.4f}",
                        'Recall': f"{entry['metrics']['recall']:.4f}",
                        'Training Size': entry['metrics']['samples_trained'],
                        'Algorithm': entry['metrics'].get('model_type', 'RandomForest'),
                        'Registry Date': entry['timestamp']
                    })
                
                df_models = pd.DataFrame(metrics_data)
                st.dataframe(df_models, use_container_width=True)
                st.success("✅ Model version history read successfully.")
            else:
                st.info("No legacy models registered in history database.")
    except Exception as e:
        st.warning(f"Unable to read version history database: {e}")

# ============================================
# ASYNCHRONOUS RUNNING TELEMETRY LOOP
# ============================================
if run_live:
    # Local buffer variables
    buffer = []
    BUFFER_SIZE = 20
    sample_count = 0
    
    st.toast("📡 Launching asynchronous hardware thread...", icon="🔌")
    time.sleep(0.5)
    
    while run_live:
        try:
            # 1. Fetch data asynchronously from queue non-blocking
            data = get_live_data()
            if data is None:
                time.sleep(0.01)
                continue
                
            # Save latest raw reading for manual calibration button
            st.session_state.latest_distance = data["distance"]
            
            # Auto-drift calibration under zero load on the very first read
            if not calib_state["is_calibrated"] and data["weight"] < 0.1:
                calib_state["baseline_distance"] = data["distance"]
                calib_state["is_calibrated"] = True
                print(f"🔌 Automatic baseline calibration set to: {calib_state['baseline_distance']} cm")
                
            sample_count += 1
            buffer.append(data)
            if len(buffer) > BUFFER_SIZE:
                buffer.pop(0)
                
            # Keep loading states clean
            if len(buffer) < BUFFER_SIZE:
                status_label_placeholder.warning(f"⏳ Dynamic signal buffers filling... ({len(buffer)}/{BUFFER_SIZE})")
                time.sleep(0.05)
                continue
                
            # Subtract baseline from the raw distance values in the buffer to get true deflection!
            calibrated_buffer = []
            for d in buffer:
                calibrated_d = d.copy()
                calibrated_d["distance"] = abs(d["distance"] - calib_state["baseline_distance"])
                calibrated_buffer.append(calibrated_d)
                
            # 2. Extract advanced math features from calibrated buffer
            features = extract_features_live(calibrated_buffer)
            
            # Compute dynamic physics-based Risk Score 
            risk_score = (
                0.5 * (features["max_deflection"] / 50.0) +
                0.3 * (features["vibration_std"] / 1.5) +
                0.2 * (abs(features["deflection_rate"]) / 50.0)
            )
            
            # 3. Model classification lookup (handling 5 or 7 dynamic features)
            result_ml, confidence_ml = predict(model, features, is_trained)
            
            # 4. Physics-informed safety guardrail overrides
            if result_ml is not None:
                result = result_ml
                confidence = confidence_ml
                
                # Digital twin physics override
                if risk_score > 0.65:
                    result = "DANGER"
                    confidence = max(confidence, 0.95)
                elif risk_score > 0.45 and result == "SAFE":
                    result = "WARNING"
            else:
                # Fallback to rule logic if no ML loaded
                if risk_score > 0.65:
                    result = "DANGER"
                    confidence = 0.90
                elif risk_score > 0.45:
                    result = "WARNING"
                    confidence = 0.70
                else:
                    result = "SAFE"
                    confidence = 0.85
                    
            # Closed-Loop Bidirectional Feedback Actuation
            if "last_alert_cmd" not in st.session_state:
                st.session_state.last_alert_cmd = 'S'
            
            # Send 'D' for DANGER, and 'S' to turn off alert for SAFE/WARNING
            current_cmd = 'D' if result == "DANGER" else 'S'
            if current_cmd != st.session_state.last_alert_cmd:
                send_serial_command(current_cmd)
                st.session_state.last_alert_cmd = current_cmd
                
            # 5. Maintain rolling chart history (using persistent cached global list)
            plotly_history.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Deflection (cm)": features["max_deflection"],
                "Vibration (g)": features["vibration_max"],
                "Weight (kg)": data["weight"]
            })
            if len(plotly_history) > 40:
                plotly_history.pop(0)
                
            # 6. Run physical 3D FEM structural solver to get real-time member stresses and displacements
            theoretical_deflection, elements_data = run_simulation(data["weight"], return_full_data=True)
            elements_json = json.dumps(elements_data)
            
            # Render live 3D WebGL structural digital twin using Three.js inside an HTML frame
            bridge_3d_html = draw_bridge_3d_html(features["max_deflection"], data["weight"], result, elements_json)
            with twin_svg_placeholder.container():
                st.components.v1.html(bridge_3d_html, height=240)
            
            # 7. Render dynamic text statistics card
            weight_metric.metric("Load Weight", f"{data['weight']:.2f} kg", help="Weight currently acting on the structural platform")
            deflection_metric.metric("Deck Deflection", f"{features['max_deflection']:.2f} cm", help="Vertical structural deformation offset")
            vibration_metric.metric("Peak Vibration", f"{features['vibration_max']:.3f} g", help="Maximum dynamic peak signal reading")
            confidence_metric.metric("ML Confidence", f"{confidence:.1%}", help="Classification confidence index score")
            
            # Render health alert boxes with premium custom aesthetics
            if result == "SAFE":
                status_label_placeholder.markdown(f"""
                <div style="background-color: rgba(0, 230, 118, 0.15); border: 2px solid #00E676; padding: 18px; border-radius: 12px; text-align: center;">
                    <h2 style="color: #00E676; margin: 0; font-weight: 800; font-family: 'Inter', sans-serif;">🟢 HEALTH STATE: SAFE</h2>
                    <p style="color: #ECEFF1; margin: 5px 0 0 0; font-size: 0.95rem;">Structure behaves within normal elastic parameters. No interventions required.</p>
                </div>
                """, unsafe_allow_html=True)
            elif result == "WARNING":
                status_label_placeholder.markdown(f"""
                <div style="background-color: rgba(255, 214, 0, 0.15); border: 2px solid #FFD600; padding: 18px; border-radius: 12px; text-align: center;">
                    <h2 style="color: #FFD600; margin: 0; font-weight: 800; font-family: 'Inter', sans-serif;">🟡 HEALTH STATE: WARNING</h2>
                    <p style="color: #ECEFF1; margin: 5px 0 0 0; font-size: 0.95rem;">Moderate load deformation observed. Inspect structural expansion joints.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                status_label_placeholder.markdown(f"""
                <div style="background-color: rgba(255, 23, 68, 0.15); border: 2px solid #FF1744; padding: 18px; border-radius: 12px; text-align: center;">
                    <h2 style="color: #FF1744; margin: 0; font-weight: 800; font-family: 'Inter', sans-serif;">🔴 HEALTH STATE: DANGER</h2>
                    <p style="color: #ECEFF1; margin: 5px 0 0 0; font-size: 0.95rem;">EXCESSIVE DEFORMATION / FAILURE SUSPECTED. CLOSE SPAN IMMEDIATELY.</p>
                </div>
                """, unsafe_allow_html=True)

            # 8. Render advanced Plotly historical telemetry graph overlay
            df_plot = pd.DataFrame(plotly_history)
            fig = go.Figure()
            
            # Left Y-Axis: Deflection
            fig.add_trace(go.Scatter(
                x=df_plot["Timestamp"], y=df_plot["Deflection (cm)"],
                name="Deflection (cm)",
                line=dict(color="#00d2ff", width=3.5, shape='spline'),
                mode='lines+markers',
                marker=dict(size=4)
            ))
            
            # Left Y-Axis: Vibration
            fig.add_trace(go.Scatter(
                x=df_plot["Timestamp"], y=df_plot["Vibration (g)"],
                name="Vibration (g)",
                line=dict(color="#9b51e0", width=2.5, dash='dash'),
                mode='lines'
            ))

            # Add clear threshold limit zone annotations
            fig.add_hline(y=40, line_width=1.5, line_dash="dash", line_color="#FFD600", opacity=0.7)
            fig.add_hline(y=60, line_width=1.5, line_dash="dash", line_color="#FF1744", opacity=0.7)

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=20),
                height=260,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Magnitude Value")
            )
            plotly_chart_placeholder.plotly_chart(fig, use_container_width=True, key="live_telemetry_plot")
            
            # 9. Render Spectral Advanced Panel Tab
            dom_freq_metric.metric("Dominant Resonance", f"{features['dominant_frequency']:.2f} Hz")
            spec_energy_metric.metric("Vibration Spectral Energy", f"{features['vibration_spectral_energy']:.1f} dB²")
            
            # Create a simple power spectrum density chart
            vibs = [np.sqrt(d["ax"]**2 + d["ay"]**2 + d["az"]**2) for d in buffer]
            vibs_detrend = vibs - np.mean(vibs)
            fft_mags = np.abs(np.fft.rfft(vibs_detrend))
            freqs = np.fft.rfftfreq(len(vibs), d=0.1) # 10Hz sampling assumed
            
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Bar(
                x=freqs[1:], y=fft_mags[1:],
                marker_color='#0066ff',
                name="Spectral Magnitude"
            ))
            fig_spec.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=20),
                height=260,
                xaxis=dict(title="Frequency (Hz)", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Amplitude Magnitude", gridcolor="rgba(255,255,255,0.06)")
            )
            spectrum_chart_placeholder.plotly_chart(fig_spec, use_container_width=True, key="spectral_vibration_plot")

            # Sleep briefly to align with sampling rate
            time.sleep(0.2)
            
        except KeyboardInterrupt:
            st.sidebar.info("Asynchronous polling loop interrupted.")
            break
        except Exception as e:
            # Safely log warnings to standard terminal console to keep dashboard error-free
            print(f"⚠️ Telemetry rendering tick warning: {e}")
            time.sleep(0.5)
            continue
else:
    st.info("ℹ️ Dynamic monitoring stream paused. Check 'Enable Active Monitoring' in the sidebar control HUD to resume.")
