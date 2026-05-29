import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class BirdDataset(Dataset):
    def __init__(self, csv_path, audio_dir):
        self.df = pd.read_csv(csv_path).head(1000)
        self.audio_dir = audio_dir
        self.labels = sorted(self.df["primary_label"].unique())
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = self.audio_dir + "/" + row["filename"]

        y, sr = librosa.load(audio_path, sr=32000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft= 1024)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_db = mel_db[:, :512]
        if mel_db.shape[1] < 512:
            mel_db = np.pad(mel_db, ((0, 0), (0, 512 - mel_db.shape[1])))

        x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.label_to_idx[row["primary_label"]])

        return x, label

dataset = BirdDataset("train.csv", "train_audio")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(8, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(16 * 32 * 128, 128),
    nn.ReLU(),
    nn.Linear(128, len(dataset.labels))
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
