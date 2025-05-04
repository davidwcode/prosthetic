#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver board = Adafruit_PWMServoDriver(0x40);

#define SERVOMIN  125
#define SERVOMAX  1250

int rotation_angle = 90; // 0 = left, 90 = middle, 180 = right

int angleToPulse(int ang) {  
  int pulse = map(ang, 0, 360, SERVOMIN, SERVOMAX);
  return pulse;
}


void move_hand(int thumb, int index, int middle, int ring, int pinkie) {
  board.setPWM(0, 0, angleToPulse(thumb)); // Thumb
  board.setPWM(1, 0, angleToPulse(index)); // Index
  board.setPWM(2, 0, angleToPulse(middle)); // Middle
  board.setPWM(3, 0, angleToPulse(ring)); // Ring
  board.setPWM(4, 0, angleToPulse(pinkie)); // Pinkie
}

void rotate(int direction) {
  if ((rotation_angle == 0 && direction == -90) || (rotation_angle == 180 && direction == 90)) {
    board.setPWM(5,0,angleToPulse(rotation_angle / 3 * 2));
  }
  else {
  rotation_angle += direction;
  board.setPWM(5,0,angleToPulse(rotation_angle / 3 * 2));
  }
}

void clearSerialBuffer() {
    while (Serial.available() > 0) {
        Serial.read();
    }
}

void setup() {
  board.begin();
  board.setPWMFreq(60);

  // Initialize all servos to rest
  move_hand(0,0,0,0,0);
  rotate(0);
  
  Serial.begin(9600);
  delay(5000);
}

void loop() {
  clearSerialBuffer();
  while (Serial.available() == 0) {
      // Wait for fresh input
  }
  float movement = Serial.readStringUntil('\n').toFloat();

  switch ((int)movement) {
    case 1: // Clench
      move_hand(110,170,170,170,170);
      break;
    case 2: // Triton
      move_hand(110,0,0,170,0);
      break;
    case 3: // L
      move_hand(0,0,170,170,170);
      break;
    case 4: // Surfer
      move_hand(0, 170, 170, 170, 0);
      break;
    case 8: // Rotate Right
      rotate(90);
      break;
    case 9: // Rotate Left
      rotate(-90);
      break;
    default: // Rest
      move_hand(0,0,0,0,0);
      break;
  }
  
}