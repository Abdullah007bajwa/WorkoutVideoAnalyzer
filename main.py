from preprocessing.data_processing import unzip_data, rename_and_copy_files, remove_duplicates
from preprocessing.split_data import split_data
from preprocessing.landmark_extraction import extract_landmarks, process_images
from models.model import build_model, train_and_save_model
from utils.video_classification import classify_video
import os
import matplotlib.pyplot as plt
import numpy as np

zip_file_path = r'C:\Users\HP\Documents\CVV\archive_16.zip' 
extracted_dir = r'data\workoutfitness'
download_dir = r'data\working\workout_images'
train_dir = r'data\working\train'
val_dir = r'data\working\val'
test_dir = r'data\working\test'
def main():


    print("[INFO] Unzipping data...")
    unzip_data(zip_file_path, extracted_dir)
    print("[INFO] Data unzipped.")

    print("[INFO] Renaming and copying files...")
    rename_and_copy_files(extracted_dir, download_dir)
    print("[INFO] Files renamed and copied.")

    print("[INFO] Removing duplicate images...")
    remove_duplicates(download_dir)
    print("[INFO] Duplicate images removed.")

    print("[INFO] Splitting data into training, validation, and test sets...")
    train_paths, val_paths, test_paths = split_data(download_dir, train_dir, val_dir, test_dir)
    print("[INFO] Data split complete.")

    print("[INFO] Extracting landmarks for training data...")
    train_images, train_labels = process_images(train_paths, train_dir)
    print("[INFO] Landmarks extracted for training data.")

    print("[INFO] Extracting landmarks for validation data...")
    val_images, val_labels = process_images(val_paths, val_dir)
    print("[INFO] Landmarks extracted for validation data.")

    print("[INFO] Extracting landmarks for test data...")
    test_images, test_labels = process_images(test_paths, test_dir)
    print("[INFO] Landmarks extracted for test data.")

    plt.figure(figsize=(10, 5))
    plt.hist(train_labels, bins=len(np.unique(train_labels)), alpha=0.7, label='Train Labels')
    plt.hist(val_labels, bins=len(np.unique(val_labels)), alpha=0.7, label='Val Labels')
    plt.hist(test_labels, bins=len(np.unique(test_labels)), alpha=0.7, label='Test Labels')
    plt.legend(loc='upper right')
    plt.title('Class Distribution in Train, Validation, and Test Sets')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

    print("[INFO] Training and saving the model...")
    best_model = train_and_save_model(train_images, train_labels, val_images, val_labels)
    print("[INFO] Model trained and saved.")

    video_file_path = r'C:\Users\HP\Documents\CVV\Profile Shot Young Woman Performing Barbell Stock Footage Video (100% Royalty-free) 1103943075.webm'
    print("[INFO] Classifying video...")
    class_dict = {class_name: idx for idx, class_name in enumerate(os.listdir(train_dir))}
    classify_video(video_file_path, best_model, class_dict)
    print("[INFO] Video classification complete.")

if __name__ == "__main__":
    main()


