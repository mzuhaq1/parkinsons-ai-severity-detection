import cv2
import mediapipe as mp
import math
import time
import csv
import os
import numpy as np
import joblib
from datetime import datetime

model = joblib.load("parkinsons_severity_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())

tap_count = 0
touching = False
running = False
recording = False
out = None

distance = 0
speed = 0
score = 4   
ml_prediction = "-"

time_series = []
distance_series = []

hand_side = "U"   # U = Unknown
class_prefix = "PD"   # Parkinson
# class_prefix = "HC" # Healthy Control
participant_number = "71"

session_duration = 10
session_start_time = None
time_left = session_duration
session_finished = False
last_saved_filename = ""

final_taps = 0
final_speed = 0
final_score = 4

fourcc = cv2.VideoWriter_fourcc(*'XVID')
csv_file = "session_results.csv"

def predict_severity(features_dict):
    import pandas as pd

    # Build DataFrame from your live features
    features_df = pd.DataFrame([features_dict])

    # Force exact same column order as model training
    features_df = features_df.reindex(columns=model.feature_names_in_)

    prediction = model.predict(features_df)
    return prediction[0]

def calculate_updrs(speed_value):
    if speed_value > 3:
        return 0
    elif speed_value > 2:
        return 1
    elif speed_value > 1:
        return 2
    elif speed_value > 0.5:
        return 3
    else:
        return 4
    
    features_dict = {
        "mean_distance": mean_val,
        "std_distance": std_val,
        "cv_distance": cv_val,
        "max_distance": max_val,
        "min_distance": min_val,
        "range_distance": range_val,
        "slowing_index": slowing_index,
        "tap_count": final_taps,
        "tap_frequency": final_speed
    }

    ml_prediction = predict_severity(features_dict)    

def save_session_to_csv(timestamp, taps, speed_value, updrs_score, video_file):
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Tap Count",
                "Speed (taps/sec)",
                "UPDRS Score",
                "Recorded Video"
            ])

        writer.writerow([
            timestamp,
            taps,
            round(speed_value, 2),
            updrs_score,
            video_file
        ])

def save_features_to_csv(video_id, distances, taps, speed_value, updrs_score, ml_prediction):
    features_file = "features_dataset.csv"
    file_exists = os.path.isfile(features_file)

    dist = np.array(distances)

    mean_val = np.mean(dist) if len(dist) > 0 else 0
    std_val = np.std(dist) if len(dist) > 0 else 0
    max_val = np.max(dist) if len(dist) > 0 else 0
    min_val = np.min(dist) if len(dist) > 0 else 0
    range_val = max_val - min_val
    cv_val = std_val / mean_val if mean_val != 0 else 0

    half = len(dist) // 2
    first_half = dist[:half]
    second_half = dist[half:]

    first_mean = np.mean(first_half) if len(first_half) > 0 else 0
    second_mean = np.mean(second_half) if len(second_half) > 0 else 0
    slowing_index = first_mean - second_mean

    with open(features_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "ID",
                "mean_distance",
                "std_distance",
                "cv_distance",
                "max_distance",
                "min_distance",
                "range_distance",
                "slowing_index",
                "tap_count",
                "tap_frequency",
                "UPDRS_label",
                "ML_prediction"
            ])

        writer.writerow([
            video_id,
            round(mean_val, 2),
            round(std_val, 2),
            round(cv_val, 4),
            round(max_val, 2),
            round(min_val, 2),
            round(range_val, 2),
            round(slowing_index, 2),
            taps,
            round(speed_value, 2),
            updrs_score,
            ml_prediction
        ])

print("Starting main loop...")

while True:
    ret, frame = cap.read()
    print("Frame read:", ret)
    if not ret:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            label = handedness.classification[0].label  # "Left" or "Right"
            hand_side = "L" if label == "Left" else "R"

            h, w, _ = frame.shape
            index_tip = hand_lms.landmark[8]
            thumb_tip = hand_lms.landmark[4]

            x1, y1 = int(index_tip.x * w), int(index_tip.y * h)
            x2, y2 = int(thumb_tip.x * w), int(thumb_tip.y * h)

            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            distance = math.hypot(x2 - x1, y2 - y1)

            if running:
                if distance < 40:
                    if not touching:
                        tap_count += 1
                        touching = True
                else:
                    touching = False

                if session_start_time is not None:
                    current_time = time.time() - session_start_time
                    time_series.append(current_time)
                    distance_series.append(distance)

    if running and session_start_time is not None:
        elapsed_time = time.time() - session_start_time
        time_left = max(0, session_duration - elapsed_time)
        speed = tap_count / elapsed_time if elapsed_time > 0 else 0
        score = calculate_updrs(speed)

        if elapsed_time >= session_duration:
            running = False
            session_finished = True
            time_left = 0

            final_taps = tap_count
            final_speed = speed
            final_score = score

            dist = np.array(distance_series)

            mean_val = np.mean(dist) if len(dist) > 0 else 0
            std_val = np.std(dist) if len(dist) > 0 else 0
            max_val = np.max(dist) if len(dist) > 0 else 0
            min_val = np.min(dist) if len(dist) > 0 else 0
            range_val = max_val - min_val
            cv_val = std_val / mean_val if mean_val != 0 else 0

            half = len(dist) // 2

            first_half = dist[:half]
            second_half = dist[half:]

            first_mean = np.mean(first_half) if len(first_half) > 0 else 0
            second_mean = np.mean(second_half) if len(second_half) > 0 else 0

            slowing_index = first_mean - second_mean

            features_list = [
                mean_val,
                std_val,
                cv_val,
                max_val,
                min_val,
                range_val,
                slowing_index,
                final_taps,
                final_speed
            ]

            ml_prediction = predict_severity(features_list)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_session_to_csv(timestamp, final_taps, final_speed, final_score, last_saved_filename)

            video_id = f"{class_prefix}_{hand_side}_{participant_number}"
            save_features_to_csv(video_id, distance_series, final_taps, final_speed, final_score, ml_prediction)

            if recording and out is not None:
                recording = False
                out.release()
                out = None
                print("Recording Stopped Automatically")

            print("Session Finished and Saved to CSV")
            print("Features Saved to features_dataset.csv")
            print(f"ML Predicted Severity: {ml_prediction}")

    status = "RUNNING" if running else "PAUSED"

    cv2.putText(frame, f"Hand: {hand_side}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, f"Taps: {tap_count}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, f"Speed: {speed:.2f} taps/sec", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(frame, f"UPDRS Score: {score}", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"ML Prediction: {ml_prediction}", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(frame, f"Status: {status}", (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, f"Time Left: {int(time_left)}s", (10, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    
    if recording:
        cv2.putText(frame, "REC", (10, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if session_finished:
        cv2.rectangle(frame, (40, 300), (620, 470), (50, 50, 50), -1)
        cv2.putText(frame, "SESSION COMPLETE", (60, 335),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Final Taps: {final_taps}", (60, 375),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Final Speed: {final_speed:.2f} taps/sec", (60, 410),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Final UPDRS Score: {final_score}", (60, 445),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(
        frame,
        "S:Start  P:Pause  R:Reset  V:Record  E:Stop Rec  Q:Quit",
        (10, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )

    if recording and out is not None:
        out.write(frame)

    cv2.imshow("Parkinson Detection System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        tap_count = 0
        touching = False
        running = True
        session_finished = False
        session_start_time = time.time()
        time_left = session_duration
        speed = 0
        score = 4
        ml_prediction = "-"
        final_taps = 0
        final_speed = 0
        final_score = 4
        time_series = []
        distance_series = []
        print("Session Started")

    elif key == ord('p'):
        running = False
        print("Session Paused")

    elif key == ord('r'):
        tap_count = 0
        touching = False
        running = False
        session_finished = False
        session_start_time = None
        time_left = session_duration
        speed = 0
        score = 4
        ml_prediction = "-"
        final_taps = 0
        final_speed = 0
        final_score = 4
        time_series = []
        distance_series = []
        print("Session Reset")

    elif key == ord('v'):
        if not recording:
            video_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            out = cv2.VideoWriter(
                video_name,
                fourcc,
                20.0,
                (frame.shape[1], frame.shape[0])
            )
            recording = True
            last_saved_filename = video_name
            print(f"Recording Started: {video_name}")

    elif key == ord('e'):
        if recording:
            recording = False
            if out is not None:
                out.release()
                out = None
            print("Recording Stopped")

    elif key == ord('q'):
        break

if out is not None:
    out.release()

cap.release()
cv2.destroyAllWindows()