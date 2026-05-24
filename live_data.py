import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import random
import time
import threading
import queue

# Try to import serial and serial.tools.list_ports
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

MPU6050_ACCEL_SCALE = 16384.0

# Thread-safe queue for sharing data between background reader and main thread
data_queue = queue.Queue(maxsize=100)
reader_thread = None
stop_event = threading.Event()
USE_REAL = False  # Starts False, dynamically set True upon successful connection
FORCE_SIMULATION = False  # Allows the UI to force simulated mode manually
serial_port_name = "COM3"  # Default fallback

# Global variable for holding the active serial connection to support bidirectional control
active_serial_conn = None

def find_arduino_port():
    """Auto-detect active COM/serial ports on Windows"""
    if not HAS_SERIAL:
        return None
    
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = p.description.lower()
        hwid = p.hwid.lower()
        # Look for typical Arduino/USB-serial descriptions
        if any(term in desc or term in hwid for term in ["arduino", "ch340", "usb-to-serial", "usb serial", "cp210", "ftdi", "prolific"]):
            print(f"🔌 Auto-discovered device on port: {p.device} ({p.description})")
            return p.device
            
    # Try COM3 first since it is the default project port
    for p in ports:
        if p.device == "COM3":
            return "COM3"
            
    # Fallback to the first available COM port
    if ports:
        print(f"🔌 No obvious Arduino found, trying first available port: {ports[0].device}")
        return ports[0].device
    return None

def serial_reader_worker():
    """Background worker thread that handles serial port polling, reading, and auto-reconnection"""
    global USE_REAL, serial_port_name, active_serial_conn, FORCE_SIMULATION
    print("🚀 Asynchronous Serial Thread Started.")
    
    last_conn_attempt = 0.0
    retry_interval = 10.0  # Attempt reconnection every 10 seconds, not blocking in between
    
    while not stop_event.is_set():
        # Handle Force Simulation Override
        if FORCE_SIMULATION:
            USE_REAL = False
            if active_serial_conn:
                try:
                    active_serial_conn.close()
                except:
                    pass
                active_serial_conn = None
            
            # Generate simulated data at 10Hz
            data = simulate_data()
            try:
                if data_queue.full():
                    data_queue.get_nowait()
                data_queue.put_nowait(data)
            except queue.Full:
                pass
            time.sleep(0.1)
            continue

        if active_serial_conn is None:
            now = time.time()
            if now - last_conn_attempt > retry_interval:
                last_conn_attempt = now
                # Attempt to discover port
                discovered_port = find_arduino_port()
                port_to_try = discovered_port if discovered_port else serial_port_name
                
                if HAS_SERIAL and port_to_try:
                    try:
                        print(f"🔌 Attempting serial connection to {port_to_try} at 115200 baud...")
                        active_serial_conn = serial.Serial(port_to_try, 115200, timeout=0.1)
                        print(f"✅ Serial connected successfully on {port_to_try}")
                        USE_REAL = True
                    except Exception as e:
                        print(f"❌ Serial connection to {port_to_try} failed: {e}")
                        active_serial_conn = None
                        USE_REAL = False
                else:
                    USE_REAL = False
            else:
                # Not time to retry yet
                USE_REAL = False
        
        # If we have a connected serial port
        if active_serial_conn and active_serial_conn.is_open:
            try:
                line = active_serial_conn.readline().decode(errors="ignore").strip()
                if not line:
                    # No data available in this tick, sleep briefly and keep looping
                    time.sleep(0.01)
                    continue
                
                # Parse values: weight, distance, ax, ay, az
                values = line.split(",")
                if len(values) != 5:
                    continue
                
                try:
                    weight = float(values[0])
                    distance = float(values[1])
                    ax = float(values[2])
                    ay = float(values[3])
                    az = float(values[4])
                except ValueError:
                    continue
                
                # Arduino sends raw MPU6050 acceleration counts. Convert to g.
                if max(abs(ax), abs(ay), abs(az)) > 16:
                    ax = ax / MPU6050_ACCEL_SCALE
                    ay = ay / MPU6050_ACCEL_SCALE
                    az = az / MPU6050_ACCEL_SCALE
                
                # Sanity filters
                if distance < 0 or distance > 400:
                    continue
                if abs(ax) > 5 or abs(ay) > 5 or abs(az) > 5:
                    continue
                
                data = {
                    "weight": weight,
                    "distance": distance,
                    "ax": ax,
                    "ay": ay,
                    "az": az,
                    "is_simulated": False,
                    "timestamp": time.time()
                }
                
                # Push data to queue (drop oldest if full to avoid lag)
                try:
                    if data_queue.full():
                        data_queue.get_nowait()
                    data_queue.put_nowait(data)
                except queue.Full:
                    pass
                
            except Exception as e:
                print(f"⚠️ Error reading from serial: {e}")
                try:
                    active_serial_conn.close()
                except:
                    pass
                active_serial_conn = None
                USE_REAL = False
                time.sleep(2)
        else:
            # Simulated data fallback: put simulated values into the queue at ~10Hz (0.1s interval)
            data = simulate_data()
            try:
                if data_queue.full():
                    data_queue.get_nowait()
                data_queue.put_nowait(data)
            except queue.Full:
                pass
            time.sleep(0.1)

def simulate_data():
    """Generates realistic physical sensor readings simulating vehicle loading and dynamic vibration"""
    # Weight varies, simulating a heavy vehicle entering the bridge once in a while
    weight = random.uniform(0.0, 8.0) if random.random() > 0.15 else 0.0
    
    # Physics-based simulation: distance (deflection) increases with weight
    # Normal bridge height/distance is ~2.0 cm. Load-induced deflection can push it down.
    if weight > 0:
        base_deflection = 12.0 + weight * random.uniform(4.0, 7.0)
        # Occasionally simulate an anomaly or high stress (DANGER / WARNING states)
        if random.random() < 0.05:
            base_deflection += random.uniform(30.0, 45.0)
    else:
        base_deflection = random.uniform(1.5, 4.5)
        # Rare structural vibration spike
        if random.random() < 0.02:
            base_deflection += random.uniform(35.0, 55.0)
            
    # Vibration (std / max) scales with load weight and deflection amplitude
    vib_intensity = random.uniform(0.05, 0.2)
    if weight > 0:
        vib_intensity += weight * random.uniform(0.08, 0.15)
    if base_deflection > 40:
        vib_intensity += random.uniform(0.5, 1.3)
        
    ax = random.uniform(-vib_intensity, vib_intensity)
    ay = random.uniform(-vib_intensity, vib_intensity)
    # az includes gravity component of ~1.0g
    az = 1.0 + random.uniform(-vib_intensity, vib_intensity)
    
    return {
        "weight": round(weight, 3),
        "distance": round(base_deflection, 3),
        "ax": round(ax, 3),
        "ay": round(ay, 3),
        "az": round(az, 3),
        "is_simulated": True,
        "timestamp": time.time()
    }

def start_background_reader():
    """Start background worker thread for asynchronous serial polling"""
    global reader_thread, stop_event
    if reader_thread is None or not reader_thread.is_alive():
        stop_event.clear()
        reader_thread = threading.Thread(target=serial_reader_worker, daemon=True)
        reader_thread.start()
        print("✅ Background serial reader thread launched.")

def stop_background_reader():
    """Signals background worker thread to stop and clean up"""
    global reader_thread, active_serial_conn
    if reader_thread is not None and reader_thread.is_alive():
        stop_event.set()
        reader_thread.join(timeout=1.0)
        if active_serial_conn:
            try:
                active_serial_conn.close()
            except:
                pass
            active_serial_conn = None
        print("🛑 Background serial reader thread stopped.")

def get_live_data():
    """Pulls the latest sensor data from the asynchronous queue. Thread-safe and non-blocking."""
    # Ensure background thread is running
    start_background_reader()
    
    # Retrieve from queue
    try:
        # Non-blocking get with a small timeout to keep loops interactive
        return data_queue.get(timeout=0.02)
    except queue.Empty:
        return None

def send_serial_command(cmd_char):
    """Writes a command character over the active serial port to actuate hardware (thread-safe)"""
    global active_serial_conn
    if active_serial_conn and active_serial_conn.is_open:
        try:
            active_serial_conn.write(cmd_char.encode())
            active_serial_conn.flush()
            print(f"📡 Bidirectional Feedback: Sent command '{cmd_char}' to ESP32/Arduino.")
            return True
        except Exception as e:
            print(f"⚠️ Failed to send serial feedback command: {e}")
            return False
    return False
