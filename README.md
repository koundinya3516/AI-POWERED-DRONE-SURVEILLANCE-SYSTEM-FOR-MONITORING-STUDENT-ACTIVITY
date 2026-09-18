# AI-Powered Drone Surveillance System for Campus Safety

An AI-powered drone surveillance prototype designed for monitoring student activities in campus environments. The system combines ESP32-S3 video streaming, face recognition, and lightweight action recognition to identify and monitor potentially aggressive activities.

## Overview

The system uses a manually controlled drone equipped with an ESP32-S3 camera module to capture and stream live video over a local wireless network. The video is processed on a computing system using computer vision and machine learning techniques.

The system consists of two main AI components:

- **Face Recognition:** Uses DeepFace with the VGG-Face model to identify enrolled individuals.
- **Action Recognition:** Uses motion-based features to classify activities into three categories:
  - Normal
  - Aggressive
  - Fight

When aggressive or fight activity is detected, the system can associate the detected activity with the recognized individual.

## System Architecture

```text
                 ┌─────────────────────┐
                 │   Manually Controlled│
                 │        Drone         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    ESP32-S3 Camera  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Wireless Video     │
                 │      Stream         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Processing System  │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        ┌─────────────────┐   ┌────────────────────┐
        │ Face Recognition│   │ Action Recognition │
        │ DeepFace        │   │ Motion-based       │
        │ VGG-Face        │   │ Classification     │
        └────────┬────────┘   └──────────┬─────────┘
                 │                       │
                 └───────────┬───────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Activity Monitoring │
                  └─────────────────────┘
```
##Features
Live video streaming using ESP32-S3
Face detection and recognition
VGG-Face embeddings through DeepFace
Motion-based activity recognition
Normal / Aggressive / Fight classification
Temporal smoothing for activity predictions
Visual results and model evaluation
Lightweight processing approach suitable for real-time experimentation
Technologies Used
Hardware
Drone platform
ESP32-S3 camera module
Laptop / computing system
Mobile hotspot / local wireless network
Software
Python
OpenCV
DeepFace
VGG-Face
NumPy
Scikit-learn
Random Forest
Google Colab
Project Structure
├── README.md
├── LICENSE
├── requirements.txt
├── src/
├── dataset/
├── results/
├── Docs/
├── data/
└── models/
Model Performance

The action-recognition Random Forest experiment achieved:

Metric	Result
OOB Accuracy	83.13%
Test Accuracy	79.09%

The test evaluation was performed using a stratified train/test split.

Results

The repository contains visualizations for:

Action-recognition confusion matrix
Class-wise accuracy
Face-recognition distance analysis
Face-embedding visualization
Dataset

The action-recognition dataset was collected and prepared specifically for this project and contains three activity classes:

Normal
Aggressive
Fight
Limitations

The current prototype uses a manually controlled drone and processes the video on an external computing system.

Autonomous navigation and GPS-based operation are not implemented in the current version.

Future Work

Potential extensions include:

GPS integration
Autonomous drone navigation
Improved activity-recognition accuracy
Larger and more diverse datasets
Advanced deep-learning models
Real-time mobile notifications
Improved tracking and monitoring capabilities
Project Report

The complete B.Tech project report is available in the Docs/ directory.

Authors
D. Vamsi
B. Praneeth Koundinya
G. Harika
M. Mahendra

Department of Electronics and Communication Engineering
R.V.R. & J.C. College of Engineering (Autonomous)
Guntur, Andhra Pradesh
2026
