import random

USE_REAL = True

try:
    import serial
    ser = serial.Serial('COM3', 115200, timeout=1)
    print("✅ Serial connected")
except Exception as e:
    print("⚠️ Serial failed:", e)
    USE_REAL = False


def get_live_data():
    if USE_REAL:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            print("RAW:", line)

            # ❌ Empty line
            if not line:
                return None

            values = line.split(",")

            # ❌ Wrong format
            if len(values) != 5:
                print("⚠️ Invalid format")
                return None

            try:
                weight = float(values[0])
                distance = float(values[1])
                ax = float(values[2])
                ay = float(values[3])
                az = float(values[4])

                # 🔥 Sanity checks
                if distance < 0 or distance > 400:
                    return None

                if abs(ax) > 5 or abs(ay) > 5 or abs(az) > 5:
                    return None

                return {
                    "weight": weight,
                    "distance": distance,
                    "ax": ax,
                    "ay": ay,
                    "az": az
                }

            except ValueError:
                return None

        except Exception as e:
            print("⚠️ Error:", e)
            return None

    else:
        return simulate_data()


def simulate_data():
    return {
        "weight": random.uniform(1, 10),
        "distance": random.uniform(1, 10),
        "ax": random.uniform(0, 2),
        "ay": random.uniform(0, 2),
        "az": random.uniform(0, 2)
    }