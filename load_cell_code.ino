#include "HX711.h"
#include <Wire.h>
#include <MPU6050.h>

// 🔹 BIDIRECTIONAL CONTROL ALERT ACTUATION
#define ALERT_LED 2  // Built-in blue LED on ESP32 (or connect to pin 2 relay/buzzer)

// 🔹 LOAD CELL
#define DT 4
#define SCK 5
HX711 scale;
float calibration_factor = 420.0;

// 🔹 ULTRASONIC
#define TRIG 18
#define ECHO 19

// 🔹 MPU6050
MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize Bidirectional Alert LED
  pinMode(ALERT_LED, OUTPUT);
  digitalWrite(ALERT_LED, LOW); // Start off

  // HX711 Load Cell
  scale.begin(DT, SCK);
  scale.set_scale(calibration_factor);
  scale.tare();

  // Ultrasonic Distance Sensor
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);

  // MPU6050 Gyro/Accelerometer
  Wire.begin(21, 22);
  mpu.initialize();

  delay(2000);
}

void loop() {
  // 🔹 Closed-Loop Bidirectional Feedback Actuator
  // Checks for commands written by the Digital Twin over the active Serial port
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'D') {
      digitalWrite(ALERT_LED, HIGH);  // Turn ON physical alert siren / built-in LED (DANGER)
    } else if (cmd == 'S') {
      digitalWrite(ALERT_LED, LOW);   // Turn OFF (SAFE / WARNING)
    }
  }

  // 🔹 LOAD CELL (stable filtering)
  float rawWeight = scale.get_units(20);

  if (abs(rawWeight) < 5) rawWeight = 0;

  static float weight = 0;
  weight = 0.8 * weight + 0.2 * rawWeight;

  // 🔹 ULTRASONIC (stable distance capture)
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  long duration = pulseIn(ECHO, HIGH, 30000);
  float distance = duration * 0.034 / 2;

  if (distance <= 0 || distance > 400) distance = 400;

  // 🔹 MPU6050
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);

  // 🔥 OUTPUT FOR DIGITAL TWIN & AI (CSV FORMAT)
  Serial.print(weight);
  Serial.print(",");
  Serial.print(distance);
  Serial.print(",");
  Serial.print(ax);
  Serial.print(",");
  Serial.print(ay);
  Serial.print(",");
  Serial.println(az);

  delay(500);
}