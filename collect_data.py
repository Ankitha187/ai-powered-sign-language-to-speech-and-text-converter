import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import sys
from feature_extraction import extract_features

LABEL = input("Enter word label: ").upper()
SEQUENCE_LENGTH = 30
SAVE_PATH = f"data/raw/{LABEL}"
os.makedirs(SAVE_PATH, exist_ok=True)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 🔥 Faster Windows backend + reduced buffer
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)

print("Press Q to quit")

sequence = []
sample_count = len(os.listdir(SAVE_PATH)) + 1

while True:

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        features = extract_features(hand_landmarks.landmark)
        sequence.append(features)

        if len(sequence) == SEQUENCE_LENGTH:
            filename = f"{LABEL}_{sample_count}.npy"
            np.save(os.path.join(SAVE_PATH, filename), np.array(sequence))
            print("Saved:", filename)
            sample_count += 1
            sequence = []

    cv2.putText(frame, f"Label: {LABEL}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing camera...")
        break


cap.release()
cv2.destroyAllWindows()
sys.exit(0)