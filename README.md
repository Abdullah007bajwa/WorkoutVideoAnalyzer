---

# Exercise Video Classifier

This project is an exercise video classification application built using TensorFlow and Flask. It processes videos to classify different types of exercises using a trained deep learning model.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Classifying a Video](#classifying-a-video)
- [Acknowledgements](#acknowledgements)

## Project Overview

The Exercise Video Classifier project involves:

1. **Data Preprocessing**: Extracting, renaming, and splitting the data.
2. **Landmark Extraction**: Extracting landmarks from images for better model training.
3. **Model Training**: Building and training a TensorFlow model.
4. **Video Classification**: Using the trained model to classify exercise videos.

## Directory Structure

```
Exercise-Video-Classifier/
├── app.py                      # Flask application
├── main.py                     # Main script for data preprocessing and model training
├── preprocessing/              # Directory for data preprocessing scripts
│   ├── data_processing.py      # Script for unzipping, renaming, and removing duplicates
│   ├── split_data.py           # Script for splitting data into train/val/test sets
│   └── landmark_extraction.py  # Script for extracting landmarks from images
├── models/                     # Directory for model scripts
│   └── model.py                # Script for building, training, and saving the model
├── utils/                      # Directory for utility scripts
│   └── video_classification.py # Script for classifying videos
├── templates/                  # Directory for Flask HTML templates
│   ├── index.html              # Homepage template
│   └── results.html            # Results page template
├── static/                     # Directory for static files (e.g., CSS)
│   └── style.css               # Stylesheet
├── uploads/                    # Directory for uploaded videos
├── data/                       # Directory for data
│   ├── workoutfitness/         # Extracted data
│   └── working/                # Directory for processed data
├── archive_16.zip              # Zipped dataset
└── README.md                   # This README file
```

## Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Steps

1. **Clone the repository**:

   ```sh
   git clone https://github.com/Abdullah007bajwa/WorkoutVideoAnalyzer.git
   cd WorkoutVideoAnalyzer
   ```

2. **Create a virtual environment**:

   ```sh
   python -m venv your-env-name
   ```

3. **Activate the virtual environment**:

   - **Windows**:

     ```sh
     your-env-name\Scripts\activate
     ```

   - **macOS/Linux**:

     ```sh
     source your-env-name/bin/activate
     ```

4. **Install the required packages**:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Flask Application

1. **Start the Flask application**:

   ```sh
   python app.py
   ```

2. **Open your web browser and navigate to**:

   ```
   http://127.0.0.1:5000
   ```

3. **Upload a video file** in one of the allowed formats (`mp4`, `webm`, `avi`).

4. **View the classification results** on the results page.

### Training the Model

To train the model, run the `main.py` script:

```sh
python main.py
```

This script will:

1. Unzip and preprocess the data.
2. Split the data into training, validation, and test sets.
3. Extract landmarks from the images.
4. Train and save the model.

### Classifying a Video

The `app.py` script handles video classification by:

1. Loading the trained model.
2. Uploading the video file through the web interface.
3. Classifying the video using the trained model.
4. Displaying the classification results.

## Acknowledgements

- TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Flask: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
