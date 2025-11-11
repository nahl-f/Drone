# preprocess_train.py
import os
import glob
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
import time
import json

# -------------------------
# USER CONFIG
# -------------------------
WAV_DIR = r"C:\Users\nahlf\Desktop\Drone\WAV"    # folder with your .wav files
SPEC_DIR = r"C:\Users\nahlf\Desktop\Drone\samples"  # output folder for .npy spectrograms
MODEL_DIR = r"C:\Users\nahlf\Desktop\Drone\MODEL" # where model + artifacts saved
os.makedirs(SPEC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Spectrogram params (must match in inference)
SR = 16000
N_MELS = 128
FMAX = 8000
N_FFT = 2048
HOP_LENGTH = 512

# Target number of time frames in spectrogram (makes fixed-size inputs)
TARGET_FRAMES = 128   # -> final shape: (1, N_MELS, TARGET_FRAMES)

# Training params
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20
VAL_SPLIT = 0.1
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# -------------------------
# 1) Convert WAV -> mel spectrogram .npy files
# -------------------------
def wav_to_mel_npy(wav_path, out_path):
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y[:,0]
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    # compute mel spectrogram (power)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX)
    S_dB = librosa.power_to_db(S, ref=np.max)  # shape: (N_MELS, frames)
    # ensure dtype float32
    S_dB = S_dB.astype(np.float32)
    np.save(out_path, S_dB)

# Process all wav files
wav_files = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")))
if not wav_files:
    raise SystemExit("No wav files found in WAV_DIR")

print(f"Converting {len(wav_files)} WAVs to Mel spectrograms in {SPEC_DIR} ...")
manifest = []
for wav in tqdm(wav_files):
    fname = os.path.basename(wav)
    label = 1 if fname.lower().startswith("drone") else 0
    out_name = os.path.splitext(fname)[0] + ".npy"
    out_path = os.path.join(SPEC_DIR, out_name)
    if not os.path.exists(out_path):  # skip if already exists
        wav_to_mel_npy(wav, out_path)
    manifest.append({"wav": wav, "spec": out_path, "label": int(label)})

# Save manifest
manifest_path = os.path.join(MODEL_DIR, "manifest.npy")
np.save(manifest_path, manifest)
with open(os.path.join(MODEL_DIR, "manifest.json"), "w") as f:
    json.dump(manifest[:1000], f, indent=2)  # small sample for inspection
print("Saved sample manifest to MODEL_DIR")

# -------------------------
# 2) Dataset + DataLoader
# -------------------------
class SpecDataset(Dataset):
    def __init__(self, manifest, target_frames=TARGET_FRAMES, augment=False):
        self.items = manifest
        self.target_frames = target_frames
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def _pad_or_trim(self, S):
        # S shape: (n_mels, frames)
        n_mels, frames = S.shape
        if frames < self.target_frames:
            pad_width = self.target_frames - frames
            S = np.pad(S, ((0,0),(0,pad_width)), mode='constant', constant_values=(S.min(),))
        elif frames > self.target_frames:
            # random crop during training
            start = 0
            if self.augment:
                start = np.random.randint(0, frames - self.target_frames + 1)
            S = S[:, start:start + self.target_frames]
        return S

    def __getitem__(self, idx):
        rec = self.items[idx]
        S = np.load(rec["spec"])  # shape (n_mels, frames)
        S = self._pad_or_trim(S)
        # normalize per-sample
        S = (S - S.mean()) / (S.std() + 1e-6)
        # convert to shape (1, n_mels, frames)
        S = np.expand_dims(S, 0).astype(np.float32)
        label = np.int64(rec["label"])
        return S, label

# create dataset
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

manifest_all = np.load(manifest_path, allow_pickle=True)
manifest_all = list(manifest_all)

dataset = SpecDataset(manifest_all, augment=True)
total = len(dataset)
val_size = int(total * VAL_SPLIT)
train_size = total - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
print(f"Total samples: {total}, train: {len(train_ds)}, val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# -------------------------
# 3) Define CNN model
# -------------------------
class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

model = SmallCNN().to(DEVICE)
print(model)

# -------------------------
# 4) Training loop
# -------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
best_val_acc = 0.0
best_model_path = os.path.join(MODEL_DIR, "best_model.pt")

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    model.train()
    train_losses = []
    preds = []
    trues = []
    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True).float()
        optimizer.zero_grad()
        out = model(xb).squeeze(1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        preds += (torch.sigmoid(out).detach().cpu().numpy() > 0.5).astype(int).tolist()
        trues += yb.cpu().numpy().astype(int).tolist()
    train_acc = accuracy_score(trues, preds)
    train_loss = float(np.mean(train_losses))

    # validation
    model.eval()
    val_preds = []
    val_trues = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True).float()
            out = model(xb).squeeze(1)
            val_preds += (torch.sigmoid(out).cpu().numpy() > 0.5).astype(int).tolist()
            val_trues += yb.cpu().numpy().astype(int).tolist()
    val_acc = accuracy_score(val_trues, val_preds)

    elapsed = time.time() - t0
    print(f"Epoch {epoch}/{EPOCHS} — train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, time: {elapsed:.1f}s")

    # checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "cfg": {
                "SR": SR, "N_MELS": N_MELS, "FMAX": FMAX, "HOP_LENGTH": HOP_LENGTH, "N_FFT": N_FFT,
                "TARGET_FRAMES": TARGET_FRAMES
            }
        }, best_model_path)
        print("Saved best model ->", best_model_path)

print("Training complete. Best val acc:", best_val_acc)
