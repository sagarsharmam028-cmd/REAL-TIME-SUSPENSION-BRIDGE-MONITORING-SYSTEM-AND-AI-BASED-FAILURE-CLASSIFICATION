#include "HX711.h"
#include <Wire.h>
#include <MPU6050.h>

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

  // HX711
  scale.begin(DT, SCK);
  scale.set_scale(calibration_factor);
  scale.tare();

  // Ultrasonic
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);

  // MPU6050
  Wire.begin(21, 22);
  mpu.initialize();

  delay(2000);
}

void loop() {

  // 🔹 LOAD CELL (stable)
  float rawWeight = scale.get_units(20);

  if (abs(rawWeight) < 5) rawWeight = 0;

  static float weight = 0;
  weight = 0.8 * weight + 0.2 * rawWeight;

  // 🔹 ULTRASONIC (stable)
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

  // 🔥 OUTPUT FOR AI (CSV FORMAT)
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