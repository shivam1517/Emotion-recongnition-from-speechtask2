import librosa
import numpy as np
import pandas as pd
import os

dataset_path = "dataset"

data = []

def extract_features(file):
    audio, sample_rate = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            emotion = root.split("/")[-1]
            file_path = os.path.join(root, file)

            features = extract_features(file_path)
            row = list(features)
            row.append(emotion)

            data.append(row)

df = pd.DataFrame(data)

df.to_csv("emotion_dataset.csv", index=False)

print("Dataset created successfully")