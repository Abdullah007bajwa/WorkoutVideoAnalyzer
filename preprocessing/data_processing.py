import zipfile
from pathlib import Path
import shutil
import os
import cv2
import numpy as np
from imutils import paths

def unzip_data(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def rename_and_copy_files(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for subdir in src_dir.iterdir():
        if subdir.is_dir():
            (dst_dir / subdir.name).mkdir(exist_ok=True)
            for file in subdir.iterdir():
                if file.is_file():
                    new_name = file.name.split('_', 1)[1]
                    new_path = dst_dir / subdir.name / new_name
                    shutil.copy(file, new_path)

def dhash(image, hashSize=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def remove_duplicates(image_dir):
    image_paths = list(paths.list_images(image_dir))
    hashes = {}

    for image_path in image_paths:
        image = cv2.imread(image_path)
        h = dhash(image)
        p = hashes.get(h, [])
        p.append(image_path)
        hashes[h] = p

    for (h, hashed_paths) in hashes.items():
        if len(hashed_paths) > 1:
            for p in hashed_paths[1:]:
                os.remove(p)
