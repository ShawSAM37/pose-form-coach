/*
 * formcoach — hardware feedback sketch
 *
 * Receives a lowercase form-quality class over serial (newline-terminated)
 * and drives the feedback hardware:
 *
 *   Class       | RGB LED | Buzzer  | Servo
 *   ------------|---------|---------|-------
 *   up          | green   | off     | 0 deg
 *   down        | blue    | off     | 0 deg
 *   optimal     | cyan    | off     | 0 deg
 *   suboptimal  | yellow  | 500 Hz  | 135 deg
 *   dangerous   | red     | 1000 Hz | 180 deg
 *
 * Wiring:
 *   Pin 7  -> servo signal
 *   Pin 8  -> piezo buzzer (+)
 *   Pin 10 -> RGB LED red   (via resistor)
 *   Pin 11 -> RGB LED green (via resistor)
 *   Pin 12 -> RGB LED blue  (via resistor)
 *
 * Comparisons are case-insensitive so the sketch also tolerates
 * capitalized labels from older client scripts.
 */

#include <Servo.h>

const int servoPin = 7;
const int buzzerPin = 8;
const int redPin = 10;
const int greenPin = 11;
const int bluePin = 12;

Servo myServo;

void setup() {
  Serial.begin(9600);

  // Short timeout keeps the loop responsive if a newline is missed;
  // the default 1000 ms stalls the hardware for a full second.
  Serial.setTimeout(10);

  pinMode(buzzerPin, OUTPUT);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);

  myServo.attach(servoPin);
  myServo.write(0);
}

void loop() {
  if (Serial.available() > 0) {
    String poseData = Serial.readStringUntil('\n');
    poseData.trim();

    // Reset outputs before applying the new state
    noTone(buzzerPin);
    setColor(0, 0, 0);

    if (poseData.equalsIgnoreCase("up")) {
      setColor(0, 255, 0);
      myServo.write(0);
    }
    else if (poseData.equalsIgnoreCase("down")) {
      setColor(0, 0, 255);
      myServo.write(0);
    }
    else if (poseData.equalsIgnoreCase("optimal")) {
      setColor(0, 255, 255);
      myServo.write(0);
    }
    else if (poseData.equalsIgnoreCase("suboptimal")) {
      setColor(255, 255, 0);
      tone(buzzerPin, 500);
      myServo.write(135);
    }
    else if (poseData.equalsIgnoreCase("dangerous")) {
      setColor(255, 0, 0);
      tone(buzzerPin, 1000);
      myServo.write(180);
    }
  }
}

// RGB values range 0 (off) to 255 (full brightness)
void setColor(int redValue, int greenValue, int blueValue) {
  analogWrite(redPin, redValue);
  analogWrite(greenPin, greenValue);
  analogWrite(bluePin, blueValue);
}
