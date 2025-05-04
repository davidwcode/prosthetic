# EMG Prosthetic Hand

**Project VADER** is an open-source, low-cost prosthetic hand system developed by Triton NeuroTech at UC San Diego for the 2025 California Neurotechnology Conference (CNTC). The system uses surface electromyography (sEMG) signals and gyroscope data to classify hand gestures and wrist rotations in real-time using hyperdimensional computing (HDC), enabling intuitive control of a 3D-printed prosthetic arm. Our system prioritizes accessibility, using readily available components to enable wider adoption in both educational and clinical contexts. Future work will focus on expanding the gesture set, improving classification robustness across users, and creating a portable, smaller-profile setup.

## Features

- **Gesture classification** using 4-channel sEMG
- **Rotation detection** using gyroscope angular velocity
- **Few-shot learning** with a centroid-based HDC model
- **Live prosthetic control** via Arduino
- **Interactive GUI** for data collection, training, and inference

## Demo Video

[![Watch the video](https://img.youtube.com/vi/IQsXgWLdUmY/0.jpg)](https://www.youtube.com/watch?v=IQsXgWLdUmY)

Live demo from the 2025 California Neurotechnology Conference (CNTC) at UC San Diego.


## 🔧 Hardware Components

- Arduino Uno
- PCA9685 16-channel Servo Driver
- Servos + springs + 200 lb fishing line
- [Inmoov 3D-printed hand](https://inmoov.fr/hand-i2)
- [Mindrove 4 channel sEMG armband](https://mindrove.com/armband/?srsltid=AfmBOopWQQx64mu99t9k1P3LsEALoqYTAx9bPntEyNLhS9PTzZek5lZo)

## Software Overview

- Real-time signal acquisition via [Boardshim](https://docs.mindrove.com/main/index.html)
- Bandpass and notch filtering for sEMG
- Classification using [torchhd](https://github.com/hyperdimensional-computing/torchhd)
- GUI for live demo of calibration, training, and inference
- Serial communication with Arduino for servo control

## Getting Started

### 1. Install Dependencies

Navigate to the `/ml` directory and install the requirements:

```bash
cd ml
pip install -r requirements.txt
```

### 2. Setup Hardware
In `main.py` (or `main4.py` if you only want 4 classes instead of 5), set the correct serial port for the Arduino. You can find the port in the Arduino IDE under Tools > Port.

### 3. Run the System
From the `/ml` directory, run the main script:

```bash
python main.py
```
or

```bash
python main4.py
```
This will start the GUI for data collection, training, and inference. Follow the on-screen instructions to collect data, train the model, and control the prosthetic hand.
