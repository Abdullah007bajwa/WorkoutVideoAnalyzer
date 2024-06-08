import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        landmarks = np.array(landmarks).flatten()
        landmarks /= np.linalg.norm(landmarks)
        return landmarks
    else:
        return None

def process_images(image_paths, data_path):
    exercise_images = []
    labels = []
    classes = os.listdir(data_path)
    class_dict = {class_name: idx for idx, class_name in enumerate(classes)}

    for image_path in image_paths:
        landmarks = extract_landmarks(image_path)
        if landmarks is not None:
            class_name = Path(image_path).parent.name
            exercise_images.append(landmarks)
            labels.append(class_dict[class_name])

    return np.array(exercise_images), np.array(labels)
