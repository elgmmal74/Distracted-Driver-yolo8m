# Distracted Driver Monitoring System using YOLOv8m

This project presents a real-time distracted driver monitoring system built using the YOLOv8m object detection model. The system is designed to monitor drivers inside the vehicle, detect unsafe behaviors, and issue alerts when necessary — helping reduce accidents caused by human distraction.

---

## Overview

Driver distraction is a major cause of road accidents. This AI-based system aims to detect and classify different types of distractions and fatigue in real time using a camera feed. Once a distraction is detected, the system can be integrated to trigger alerts or notifications to warn the driver.

---

## Key Features

- Real-time driver behavior monitoring
- Detection of multiple distraction types
- Easy deployment with a webcam or dashcam
- Can be integrated with in-car alert systems (sound, vibration, etc.)
- High accuracy model trained on real-world data

---

## Detected Classes

The model is trained to recognize the following behaviors:

- Safe Driving  
- Texting  
- Talking on the Phone  
- Drinking  
- Reaching Behind  
- Talking to Passenger  
- Eyes Closed  
- Yawning  
- Nodding Off

---

## Model Details

- **Model Used**: YOLOv8m (Ultralytics)
- **Input Size**: 640x640
- **Frameworks**: Python, OpenCV, Ultralytics YOLO
- **Accuracy**: Achieved high accuracy on the validation dataset with excellent performance across most classes, as shown in the confusion matrix.

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/elgmmal74/Distracted-Driver-yolo8m
cd Distracted-Driver-yolo8m
