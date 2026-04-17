# Parkinson’s AI Severity Detection System

## Overview
This project uses Computer Vision and Machine Learning to analyse hand movement patterns and predict Parkinson’s disease severity based on UPDRS scores.

## System Pipeline
Video Input → Hand Tracking (MediaPipe / YOLO) → Feature Extraction → Machine Learning Model → Severity Prediction

## Features Extracted
- Mean Distance
- Standard Deviation
- Coefficient of Variation
- Max/Min Distance
- Range
- Slowing Index
- Tap Count
- Tap Frequency

## Machine Learning Model
- Model: Random Forest Classifier
- Training/Test Split: 80/20
- Accuracy: **0.95**
- Weighted F1 Score: **0.95**

## Evaluation
- Confusion Matrix used for performance analysis
- Strong classification across all UPDRS levels
- Minor misclassification between adjacent severity levels

## Data Leakage Prevention
GroupShuffleSplit was used to ensure that samples from the same participant (ID) do not appear in both training and testing datasets.

## Demo Application
- Real-time hand tracking
- Feature extraction from finger tapping movement
- Live prediction of Parkinson’s severity

## Project Structure

data/ → datasets
model/ → trained ML model
results/ → evaluation outputs
src/ → main scripts

## How to Run

### Install dependencies

pip install -r requirements.txt

### Train model

python src/train_model.py

### Run demo

python src/parkinson_detection_demo.py

## Author
Muhammad Zain Ul Haq

