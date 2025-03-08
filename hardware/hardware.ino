#include <Adafruit_PWMServoDriver.h>

// Adafruit PWM Servo Driver object with I2C address 0x40
Adafruit_PWMServoDriver board = Adafruit_PWMServoDriver(0x40);

#define SERVOMIN  125
#define SERVOMAX  512

// Convert angle to PWM pulse width
int angleToPulse(int ang) {  
  int pulse = map(ang, 0, 180, SERVOMIN, SERVOMAX);
  return pulse;
}

void clearSerialBuffer() {
  while (Serial.available() > 0) {
      Serial.read();  // Discard old bytes
  }
}

void move_hand(int pinkie, int ring, int middle, int index, int thumb) {
  board.setPWM(0, 0, angleToPulse(pinkie)); // Pinkie
  board.setPWM(1, 0, angleToPulse(ring)); // Ring
  board.setPWM(2, 0, angleToPulse(middle)); // Middle
  board.setPWM(3, 0, angleToPulse(index)); // Index
  board.setPWM(4, 0, angleToPulse(thumb)); // Thumb
}

void setup() {
  // Initialize PWM servo driver
  board.begin();
  board.setPWMFreq(60); // Set PWM frequency to 60 Hz

  // Initialize all servos to rest
  move_hand(180,180,0,180,180);
  
  Serial.begin(9600); // Start serial communication
  delay(5000); // Give time for servos to settle
}

void loop() {


  clearSerialBuffer();
  while (Serial.available() == 0) {
      // Wait for fresh input
  }
  float movement = Serial.readStringUntil('\n').toFloat();

  if (movement == 1) {
    move_hand(0,0,180,0,0);
  }
  else if (movement == 2) {
    move_hand(180,0,180,180,0);
  }
  else if (movement == 3) {
    move_hand(0,0,180,180,180);
  }
  else {
    move_hand(180,180,0,180,180);
  }

}
