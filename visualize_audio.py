import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("train.csv")

# Take first row
row = df.iloc[0]

# Build full audio path
audio_path = "train_audio/" + row["filename"]

print("Audio path:", audio_path)
print("Label:", row["primary_label"])

# Load audio
y, sr = librosa.load(audio_path, sr=32000)

# Create mel spectrogram
mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel_db = librosa.power_to_db(mel, ref=np.max)

# Plot
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mel_db,
    sr=sr,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title(row["primary_label"])

plt.tight_layout()
plt.show()