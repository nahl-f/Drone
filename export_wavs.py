import os
import pandas as pd
import soundfile as sf
import io
from glob import glob
from tqdm import tqdm

# --- Paths ---
DATA_DIR = r"C:\Users\nahlf\Desktop\Drone\drone-audio-detection-samples\data"   # where parquet files are
OUTPUT_DIR = r"C:\Users\nahlf\Desktop\Drone\WAV"  # your target WAV folder

# --- Create output folder if it doesn't exist ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Gather parquet files ---
parquet_files = sorted(glob(os.path.join(DATA_DIR, "*.parquet")))

if not parquet_files:
    print("⚠️ No parquet files found in:", DATA_DIR)
    exit()

# --- Counters for naming ---
drone_count = 0
background_count = 0

# --- Process each parquet file ---
for parquet_path in tqdm(parquet_files, desc="Processing Parquet Files"):
    df = pd.read_parquet(parquet_path)
    
    for _, row in df.iterrows():
        audio_bytes = row["audio"]["bytes"]
        label = row["label"]

        # Decode WAV bytes
        y, sr = sf.read(io.BytesIO(audio_bytes))

        # Determine label and filename
        if label == 1:
            drone_count += 1
            filename = f"drone_{drone_count:04d}.wav"
        else:
            background_count += 1
            filename = f"background_{background_count:04d}.wav"

        # Full save path
        save_path = os.path.join(OUTPUT_DIR, filename)

        # Write WAV
        sf.write(save_path, y, sr)

print(f"\n✅ Export complete!")
print(f"Drone samples saved: {drone_count}")
print(f"Background samples saved: {background_count}")
print(f"Files saved in: {OUTPUT_DIR}")
