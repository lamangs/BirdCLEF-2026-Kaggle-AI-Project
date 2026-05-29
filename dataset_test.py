import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self, csv_path, audio_dir):
        self.df = pd.read_csv(csv_path).head(20)
        self.audio_dir = audio_dir
        self.labels = sorted(self.df["primary_label"].unique())
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = self.audio_dir + "/" + row["filename"]

        y, sr = librosa.load(audio_path, sr=32000)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        y_label = torch.tensor(self.label_to_idx[row["primary_label"]])

        return x, y_label

dataset = BirdDataset("train.csv", "train_audio")

x, y = dataset[0]

print("Spectrogram tensor shape:", x.shape)
print("Label:", y)