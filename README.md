# 🏠 AI-Based Gesture Controlled Smart Home Automation System with Face Authentication

An AI-powered Smart Home Automation System that combines **Face Authentication**, **Hand Gesture Recognition**, and **IoT** to provide secure, touchless control of household appliances. The project uses **Computer Vision** for user authentication and gesture recognition, while an **ESP32** microcontroller communicates with the **Blynk Cloud** to control smart devices in real time.

---

## 📖 Project Overview

Traditional home automation systems rely on physical switches or mobile applications, which can be inconvenient and lack strong security. This project introduces a secure and contactless alternative by integrating Artificial Intelligence, Computer Vision, and IoT technologies.

The system first authenticates the user using **OpenCV Face Recognition**. After successful authentication, **MediaPipe** detects predefined hand gestures through a webcam. These gestures are interpreted as commands and sent to the **ESP32** via the **Blynk IoT Platform**, allowing users to control appliances such as lights, fans, and sockets without physical interaction.

The solution offers enhanced security, real-time performance, and a scalable architecture suitable for future smart home applications.

---

## ✨ Features

- 🔐 Face Authentication
- ✋ Real-Time Hand Gesture Recognition
- 🏠 Smart Home Automation
- 📡 ESP32 Wi-Fi Communication
- ☁️ Blynk IoT Cloud Integration
- ⚡ Real-Time Device Control
- 👤 Secure User Access
- 🖥️ Computer Vision-Based Automation
- 📱 Remote Monitoring (Blynk App)
- 🔄 Scalable System Architecture

---

## 🛠️ Technologies Used

### Programming
- Python
- C++ (ESP32)

### Computer Vision
- OpenCV
- MediaPipe

### Hardware
- ESP32 Development Board
- Relay Module
- Webcam
- Home Appliances

### IoT
- Blynk Cloud
- Wi-Fi

### Development Tools
- VS Code
- Arduino IDE

---

## ⚙️ System Architecture

```
              Webcam
                 │
                 ▼
      Face Authentication
          (OpenCV)
                 │
        Authorized User
                 │
                 ▼
      Hand Gesture Detection
          (MediaPipe)
                 │
                 ▼
      Gesture Recognition
                 │
                 ▼
         Python Application
                 │
                 ▼
          Blynk IoT Cloud
                 │
                 ▼
             ESP32 Board
                 │
                 ▼
         Relay Module
                 │
                 ▼
     Lights • Fan • Socket • TV
```

---

## 🔄 Workflow

1. Capture user's face using webcam.
2. Verify identity using Face Authentication.
3. Detect hand gestures using MediaPipe.
4. Recognize gesture commands.
5. Send commands to ESP32 via Blynk Cloud.
6. ESP32 controls connected appliances.
7. Device status updates in real time.

---

## 📂 Project Structure

```
AI-Smart-Home/
│
├── data/
│   ├── known_faces/
│   └── gestures/
│
├── face_auth/
│   ├── train.py
│   ├── recognize.py
│   └── dataset.py
│
├── gesture_control/
│   ├── detector.py
│   ├── recognizer.py
│   └── commands.py
│
├── esp32/
│   └── smart_home.ino
│
├── config/
│   └── config.py
│
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/AI-Smart-Home.git

cd AI-Smart-Home
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Environment

Windows

```powershell
.\.venv\Scripts\activate
```

Linux/Mac

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 📋 Requirements

```
opencv-python
mediapipe
numpy
face-recognition
blynklib
pyserial
```

Install using

```bash
pip install -r requirements.txt
```

---

## 💡 Future Scope

- Voice Assistant Integration
- Mobile Dashboard
- Multiple User Profiles
- Cloud Database
- AI Gesture Learning
- Energy Monitoring
- Home Security Alerts
- Smart Appliance Scheduling

---

## 📊 Project Highlights

- Artificial Intelligence
- Computer Vision
- Face Authentication
- Hand Gesture Recognition
- IoT Automation
- ESP32 Integration
- Real-Time Communication
- Secure Smart Home Control

---

## 🎯 Applications

- Smart Homes
- Elderly Assistance
- Healthcare
- Offices
- Hotels
- Laboratories
- Contactless Automation
- Home Security

---

## 👨‍💻 Developed By

**Neha Yadav**

---

## 📜 License

This project is developed for educational and research purposes.
