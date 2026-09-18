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
## Features
Live video streaming using ESP32-S3.<br>
Face detection and recognition.<br>
VGG-Face embeddings through DeepFace.<br>
Motion-based activity recognition.<br>
Normal / Aggressive / Fight classification.<br>
Temporal smoothing for activity predictions.<br>
Visual results and model evaluation.<br>
Lightweight processing approach suitable for real-time experimentation.
## Technologies Used
### Hardware
Drone platform.<br>
ESP32-S3 camera module.<br>
Laptop / computing system.<br>
Mobile hotspot / local wireless network.<br>
### Software
Python,
OpenCV,
DeepFace,
VGG-Face,
NumPy,
Scikit-learn,
Random Forest,
Google Colab.
## Project Structure
├── README.md<br>
├── LICENSE<br>
├── requirements.txt<br>
├── src/<br>
├── dataset/<br>
├── results/<br>
├── Docs/<br>
├── data/<br>
└── models/<br>
## Model Performance

### The action-recognition Random Forest experiment achieved:

Metric	Result
OOB Accuracy	83.13%<br>
Test Accuracy	79.09%<br>

The test evaluation was performed using a stratified train/test split.

## Results

### The repository contains visualizations for:

Action-recognition confusion matrix.<br>
Class-wise accuracy.<br>
Face-recognition distance analysis.<br>
Face-embedding visualization<br>
## Dataset

### The action-recognition dataset was collected and prepared specifically for this project and contains three activity classes:

Normal<br>
Aggressive<br>
Fight<br>
## Limitations

1) The current prototype uses a manually controlled drone and processes the video on an external computing system.<br>

2) Autonomous navigation and GPS-based operation are not implemented in the current version.<br>

## Future Work

### Potential extensions include:

1)GPS integration.<br>
2)Autonomous drone navigation.<br>
3)Improved activity-recognition accuracy.<br>
4)Larger and more diverse datasets.<br>
5)Advanced deep-learning models.<br>
6)Real-time mobile notifications.<br>
7)Improved tracking and monitoring capabilities.<br>

## Project Report

The complete B.Tech project report is available in the Docs/ directory.

## Authors
**D. Vamsi**<br>
**B. Praneeth Koundinya**<br>
**G. Harika**<br>
**M. Mahendra**<br>

**Department of Electronics and Communication Engineering**<br>
**R.V.R. & J.C. College of Engineering (Autonomous)**<br>
Guntur, Andhra Pradesh
2026
