
# Automated Traffic Light Control and Violation Detection System

## Project Overview

This project aims to develop an automated system that optimizes traffic light control based on vehicle density and detects traffic violations, such as line violations and non-helmet wearing riders. The system leverages image processing, machine learning, and real-time video feeds to ensure efficient traffic management and road safety. Key technologies include OpenCV, YOLOv3, SVM, and ARIMA for traffic density prediction and violation detection.

## Table of Contents
1. [Project Setup](#project-setup)
2. [System Features](#system-features)
3. [Modules](#modules)
4. [Requirements](#requirements)
5. [Running the Application](#running-the-application)
6. [Future Work](#future-work)

---

## Project Setup

1. Clone the project repository.
2. Ensure that you have the necessary software libraries and packages installed (see Requirements section).
3. Set up the camera and hardware according to the specification provided in the hardware requirements.
4. Connect the camera feed to the system and configure the inputs for traffic video or images.

---

## System Features

1. **Automated Traffic Light Control**: 
   - Controls the traffic lights based on the vehicle density using machine learning techniques (SVM and ARIMA).
   - Real-time prediction of traffic density using captured frames from video feeds.
   
2. **Violation Detection**:
   - **Helmet Detection**: Detects vehicles where the rider is not wearing a helmet.
   - **Line Violation Detection**: Identifies vehicles that cross traffic lines during red lights.
   - **License Plate Recognition**: Extracts license plate information for vehicles that commit violations using OCR.

3. **Real-time Monitoring**:
   - The system provides real-time analysis of traffic density and detects potential violations without manual intervention.

---

## Modules

1. **Traffic Light Control**:
   - Inputs real-time video feed, processes the video frames, and calculates vehicle density.
   - The ARIMA model predicts future densities, and traffic lights are controlled accordingly. For instance, the green light duration is set based on vehicle density percentages (e.g., 0-30% density = 15s green light).

2. **Violation Detection**:
   - **Line Violation**: YOLOv3 is used to detect vehicles that cross traffic lines during a red light. 
   - **Helmet Detection**: Identifies non-helmet wearing riders and stores the images of violators for further action.
   - **License Plate Recognition**: OCR and contour detection techniques are used to extract license plate information for detected violators.

3. **Image Processing**:
   - OpenCV is used extensively for processing video feeds, image enhancement, feature extraction, and object detection.
   
---

## Requirements

### Software
- **Operating System**: Linux-based OS (Ubuntu recommended) or Raspbian for microcontrollers.
- **Languages**: Python 3.x
- **Libraries**:
   - `OpenCV`: For image and video processing.
   - `YOLOv3`: For object detection (helmet and line violation detection).
   - `numpy`, `pandas`, `scikit-learn`, `statsmodel`: For data handling and machine learning.
   - `easyOCR`: For license plate text extraction.
   - `tkinter`: For the graphical user interface.
   - `ImageIo`, `PIL`: For image manipulation.

### Hardware
- **Camera**: Minimum 12 MP, capable of capturing real-time traffic footage.
- **Processor**: Microcontroller such as Arduino or Raspberry Pi (with Python support).
- **Other**: Camera feed input, wired or wireless network connection.

---

## Running the Application

1. **Video Feed**: 
   - Start the camera feed to capture traffic junction footage in real-time.
   
2. **Image Processing and Detection**: 
   - The system will split the video into frames and apply image processing techniques to detect vehicles and violations.
   
3. **Traffic Light Control**: 
   - Based on the vehicle density, the system will automatically adjust the traffic lights.
   
4. **Violation Reporting**: 
   - In case of any violations, the system will capture an image of the vehicle, perform license plate recognition, and store the data for further processing.

---

## Future Work

- **Improved Accuracy**: Further refinement of the image processing and machine learning models to improve detection accuracy under various lighting and environmental conditions.
- **Integration with Government Databases**: Automatic reporting of violations to the relevant authorities.
- **Scalability**: Extend the system to handle multiple traffic junctions and integrate real-time traffic data from multiple sources
- **Advanced Violation Detection**: Detect additional violations such as speeding or signal jumping using more advanced AI techniques.

---

## References

- Dissertation Report: [Automated Traffic Light Control And Violation Detection System](#)
- Low Level Design Document: [Low Level Design and Implementation](#)

---

