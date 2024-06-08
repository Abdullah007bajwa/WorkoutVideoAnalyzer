import os
from pathlib import Path
import shutil
import random
from sklearn.model_selection import train_test_split
from imutils import paths

def split_data(download_dir, train_dir, val_dir, test_dir):
    def split_image_folder(image_paths, folder):
        data_path = Path(folder)
        if not data_path.is_dir():
            data_path.mkdir(parents=True, exist_ok=True)
        for path in image_paths:
            full_path = Path(path)
            image_name = full_path.name
            label = full_path.parent.name
            label_folder = data_path / label
            if not label_folder.is_dir():
                label_folder.mkdir(parents=True, exist_ok=True)
            destination = label_folder / image_name
            shutil.copy(path, destination)

    print("[INFO] Getting file paths and shuffling")
    image_paths = list(sorted(paths.list_images(download_dir)))
    random.shuffle(image_paths)

    print("[INFO] Configuring training and testing data")
    class_names = [Path(x).parent.name for x in image_paths]
    train_paths, rest_of_paths = train_test_split(image_paths, stratify=class_names, test_size=0.15, shuffle=True, random_state=42)

    class_names_ = [Path(x).parent.name for x in rest_of_paths]
    val_paths, test_paths = train_test_split(rest_of_paths, stratify=class_names_, test_size=0.50, shuffle=True, random_state=42)

    print("[INFO] Creating ImageFolder's for training and validation datasets")
    split_image_folder(train_paths, train_dir)
    split_image_folder(val_paths, val_dir)
    split_image_folder(test_paths, test_dir)

    print("[INFO] Dataset split and copying complete.")
    return train_paths, val_paths, test_paths
