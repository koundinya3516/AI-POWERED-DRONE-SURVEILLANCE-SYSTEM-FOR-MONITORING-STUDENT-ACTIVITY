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
