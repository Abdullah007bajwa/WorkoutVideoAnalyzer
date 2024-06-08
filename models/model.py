import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
import matplotlib.pyplot as plt

def build_model(hp, num_classes):
    print("[INFO] Building model...")
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=512, step=32), activation='relu', input_shape=(132,)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("[INFO] Model built successfully.")
    return model

def train_and_save_model(train_images, train_labels, val_images, val_labels):
    print("[INFO] Starting training process...")
    num_classes = len(np.unique(train_labels))  
    tuner = RandomSearch(
        lambda hp: build_model(hp, num_classes), 
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,
        directory='keras_tuner_dir',
        project_name='exercise_classifier'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("[INFO] Searching for the best model...")
    tuner.search(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels), callbacks=[early_stopping])
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save('optimized_exercise_classifier_model.h5')
    print("[INFO] Best model saved.")

 
    print(best_model.summary())

    return best_model

from tensorflow.keras.models import load_model as keras_load_model

def load_best_model(model_path):
    print("[INFO] Loading best model...")
    model = keras_load_model(model_path)
    print("[INFO] Model loaded successfully.")
    return model
