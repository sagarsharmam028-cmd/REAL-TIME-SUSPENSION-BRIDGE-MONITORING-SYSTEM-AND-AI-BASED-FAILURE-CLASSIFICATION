import serial

ser = serial.Serial('COM3', 115200)

def get_live_data():
    while True:
        try:
            line = ser.readline().decode().strip()
            values = line.split(",")

            if len(values) == 5:
                return {
                    "weight": float(values[0]),
                    "distance": float(values[1]),
                    "ax": float(values[2]),
                    "ay": float(values[3]),
                    "az": float(values[4])
                }
        except:
            continue