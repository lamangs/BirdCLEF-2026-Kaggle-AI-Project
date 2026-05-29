import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn

df = pd.read_csv("train.csv").head(1000)
labels = sorted(df["primary_label"].unique())

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(16 * 32 * 128, 128),
    nn.ReLU(),
    nn.Linear(128, len(labels))
)

model.load_state_dict(torch.load("bird_model.pth"))
model.eval()

audio_path = "train_audio/1161364/iNat1216197.ogg"

y, sr = librosa.load(audio_path, sr=32000)

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128,
    n_fft=1024
)
mel_db = librosa.power_to_db(mel, ref=np.max)

mel_db = mel_db[:, :512]

if mel_db.shape[1] < 512:
    mel_db = np.pad(mel_db, ((0, 0), (0, 512 - mel_db.shape[1])))

x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    outputs = model(x)
    probs = torch.sigmoid(outputs)[0]

top_indices = torch.topk(probs, 5).indices

for idx in top_indices:
    print(labels[idx], probs[idx].item())
