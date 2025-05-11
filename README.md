# EMG Prosthetic Hand

**Project VADER** is an open-source, low-cost prosthetic hand system developed by Triton NeuroTech at UC San Diego for the [2025 California Neurotechnology Conference (CNTC)](https://caneurotech.vercel.app). The system uses surface electromyography (sEMG) signals to classify hand gestures in real-time using hyperdimensional computing (HDC), enabling intuitive control of a 3D-printed prosthetic arm. Our system prioritizes accessibility, using readily available components to enable wider adoption in both educational and clinical contexts. Future work will focus on expanding the gesture set, improving classification robustness across users, and creating a portable, smaller-profile setup.

## Features

- **Gesture classification** using 4-channel sEMG
- **Rotation detection** using gyroscope angular velocity
- **Few-shot learning** with a centroid-based HDC model
- **Live prosthetic control** via Arduino
- **Interactive GUI** for data collection, training, and inference


## Demo Video (Youtube)

[![Watch the video](https://img.youtube.com/vi/IQsXgWLdUmY/0.jpg)](https://www.youtube.com/watch?v=IQsXgWLdUmY)

Live demonstration from the 2025 California Neurotechnology Conference (CNTC) at UC San Diego.

[Click here to view the full presentation slides (PDF)](docs/TNT%20BCI%20Comp%20CNTC%202025.pdf)


## Hardware Components

- Arduino Uno
- [Mindrove 4-channel sEMG armband](https://mindrove.com/armband/?srsltid=AfmBOopWQQx64mu99t9k1P3LsEALoqYTAx9bPntEyNLhS9PTzZek5lZo)
- [PCA9685 16-channel Servo Driver](https://www.adafruit.com/product/815)  
- [Miuzei DS3218MG High-Torque Digital Servo](https://www.amazon.com/Miuzei-Torque-Digital-Waterproof-Control/dp/B07HNTKSZT)
- Custom 3D-printed servo pulley (binary STL): [pulleyV3.stl](hardware/pulleyV3.stl)
- [InMoov 3D-printed hand](https://inmoov.fr/hand-i2)
- **Extension Springs** (3/16″x1-3/4 or 4.8mm x 44.5mm)
- [Braided Fishing Line (200 lb, 0.75 mm red)](https://www.amazon.com/dp/B0791DC7C8?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1)  


## Software Overview

- Real-time signal acquisition via [Boardshim](https://docs.mindrove.com/main/index.html)
- Bandpass and notch filtering for sEMG []
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
In order to actually control a hand you must have a PCA 9685, 5 Servos, and an Arduino. For wiring, you can follow this guide: [Wiring Guide](https://www.instructables.com/Mastering-Servo-Control-With-PCA9685-and-Arduino/). Once the Arduino is connected to your computer, make sure to flash the `hardware.ino` file which can be found under the hardware directory.

### 3. Connect Hardware
In `main.py` (or `main4.py` if you only want 4 classes instead of 5), set the correct serial port for the Arduino. You can find the port in the Arduino IDE under Tools > Port.

### 4. Run the System
From the `/ml` directory, run the main script:

```bash
python main.py
```
or

```bash
python main4.py
```
This will start the GUI for data collection, training, and inference. Follow the on-screen instructions to collect data, train the model, and control the prosthetic hand.
