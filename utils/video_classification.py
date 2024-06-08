import cv2
import numpy as np
import time
from mediapipe import solutions as mp_solutions
from pathlib import Path

def preprocess_frame(frame, pose):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        landmarks = np.array(landmarks).flatten()
        landmarks /= np.linalg.norm(landmarks)
        return landmarks
    else:
        return None

def predict_exercise(landmarks, model):
    landmarks = landmarks.reshape(1, -1)
    predictions = model.predict(landmarks)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)
    return predicted_class[0], confidence

def classify_video(video_file_path, best_model, class_dict):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    mp_pose = mp_solutions.pose.Pose()
    inverse_class_dict = {v: k for k, v in class_dict.items()}

    pTime = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        landmarks = preprocess_frame(frame, mp_pose)
        if landmarks is not None:
            predicted_class, confidence = predict_exercise(landmarks, best_model)
            exercise_name = inverse_class_dict[predicted_class]
            cv2.putText(frame, f"{exercise_name} ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, f"FPS: {int(fps)}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Exercise Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()