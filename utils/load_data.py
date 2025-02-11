import os
import numpy as np
from utils.feature_extraction import extract_feature

def load_data(data_path):
    X, y = [], []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if not file.endswith(".wav"):
                continue

            # Extract emotion label
            emotion = int(file.split("-")[2]) - 1

            # Extract features
            feature = extract_feature(file_path)
            if feature is not None:  # Skip files with errors
                X.append(feature)
                y.append(emotion)

    return np.array(X), np.array(y)