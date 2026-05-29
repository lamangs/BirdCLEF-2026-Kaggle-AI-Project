import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn

# LOAD LABELS
df = pd.read_csv("train.csv").head(500)
labels = sorted(df["primary_label"].unique())

# MODEL
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

# LOAD SAMPLE SUBMISSION
submission = pd.read_csv("sample_submission.csv")

# GET TEST FILES
test_folder = "test_soundscapes"

for i, row in submission.iterrows():

    filename = row["row_id"] + ".ogg"

    audio_path = os.path.join(test_folder, filename)

    if not os.path.exists(audio_path):
        continue

    y, sr = librosa.load(audio_path, sr=32000)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = mel_db[:, :512]

    if mel_db.shape[1] < 512:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, 512 - mel_db.shape[1]))
        )

    x = torch.tensor(
        mel_db,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.sigmoid(outputs)[0]

    for j, label in enumerate(labels):
        if label in submission.columns:
            submission.at[i, label] = probs[j].item()

submission.to_csv("submission.csv", index=False)

print("submission.csv created")