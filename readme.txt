Parkinson’s AI Detection System

This project uses Computer Vision and Machine Learning to analyse finger tapping movements and predict Parkinson’s severity (UPDRS score).

Requirements:
- Python 3.11

Setup:
1. Create virtual environment:
   py -3.11 -m venv mp_env

2. Activate:
   mp_env\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

Run:
python parkinson_detection_demo.py

Controls:
S = Start
R = Reset
V = Record
Q = Quit

Pipeline:
1. Video/Webcam Input
2. Hand Detection (YOLO + MediaPipe)
3. Feature Extraction (tap count, speed, distance)
4. Dataset Creation
5. Machine Learning Model (Random Forest)
6. Severity Prediction

Results:
- Accuracy: 95%
- F1 Score: 0.95

Files:
- train_data.xlsx → training dataset
- test_data.xlsx → testing dataset
- parkinson_model.pkl → trained model
- ml_confusion_matrix.png → evaluation