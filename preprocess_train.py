import os
import json
import random
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import soundfile as sf

# ======================
# CONFIG
# ======================
SAMPLE_RATE = 16000
CLIP_DURATION = 1.0  # seconds
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
BATCH_SIZE = 64
EPOCHS = 10
DATA_DIR = r"C:\Users\nahlf\Desktop\Drone\WAV"
OUTPUT_DIR = r"C:\Users\nahlf\Desktop\Drone\MODEL"

# ======================
# DATASET
# ======================
class DroneDataset(Dataset):
    def __init__(self, manifest_path, sample_rate=16000, clip_samples=16000):
        with open(manifest_path, "r") as f:
            self.data = json.load(f)

        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=256,
            n_mels=64
        )

    def __len__(self):
        return len(self.data)

    def _fix_length(self, waveform):
        num_samples = waveform.shape[1]
        if num_samples < self.clip_samples:
            pad_amount = self.clip_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif num_samples > self.clip_samples:
            waveform = waveform[:, :self.clip_samples]
        return waveform

    def __getitem__(self, idx):
        item = self.data[idx]
        # ✅ Use soundfile instead of torchaudio.load to avoid FFmpeg issues
        wav, sr = sf.read(item["path"])
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)  # convert to mono
        # ✅ FIX: Convert to float32 to match PyTorch's expected dtype
        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = self._fix_length(wav)
        mel = self.mel_transform(wav)
        # ✅ FIX: Use amplitude_to_DB with all required parameters
        mel = torchaudio.functional.amplitude_to_DB(
            mel, 
            multiplier=10.0, 
            amin=1e-10, 
            db_multiplier=0.0, 
            top_db=80.0
        )
        return mel, torch.tensor(item["label"], dtype=torch.long)

# ======================
# MODEL
# ======================
class DroneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate the correct flattened size dynamically
        self.flatten = nn.Flatten()
        
        # Placeholder - will be set after first forward pass
        self.fc_layers = None

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        
        # Initialize FC layers on first forward pass
        if self.fc_layers is None:
            flattened_size = x.shape[1]
            self.fc_layers = nn.Sequential(
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            ).to(x.device)
        
        return self.fc_layers(x)

# ======================
# BUILD MANIFEST
# ======================
def build_manifest():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    drone_files = [os.path.join(DATA_DIR, f) for f in files if "drone" in f.lower()]
    bg_files = [os.path.join(DATA_DIR, f) for f in files if "background" in f.lower()]

    if not drone_files or not bg_files:
        raise ValueError("No drone/background files found. Make sure filenames include 'drone' or 'background'.")

    n = min(len(drone_files), len(bg_files), 5000)
    random.seed(42)
    drone_samples = random.sample(drone_files, n)
    bg_samples = random.sample(bg_files, n)

    manifest = [{"path": f, "label": 1} for f in drone_samples] + [{"path": f, "label": 0} for f in bg_samples]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Selected {len(drone_samples)} drone and {len(bg_samples)} background samples ({len(manifest)} total).")
    print(f"Manifest saved at {manifest_path}")
    return manifest_path

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    manifest_path = build_manifest()
    dataset = DroneDataset(manifest_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DroneCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("\nStarting training...\n")
    for epoch in range(EPOCHS):
        total_loss = 0
        for mel, label in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            mel, label = mel.to(device), label.to(device)
            opt.zero_grad()
            preds = model(mel)
            loss = loss_fn(preds, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} done. Avg loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "drone_model.pth"))
    print(f"✅ Model saved to {OUTPUT_DIR}\\drone_model.pth")