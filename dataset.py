import pandas as pd
import soundfile as sf
import io
import numpy as np

# Load one Parquet file
df = pd.read_parquet("drone-audio-detection-samples/data/train-00038-of-00039.parquet")

# Take one sample
sample = df.iloc[0]
audio_bytes = sample["audio"]["bytes"]

# Decode the bytes as a WAV file
y, sr = sf.read(io.BytesIO(audio_bytes))

print("Label:", sample["label"])
print("Audio shape:", y.shape)
print("Sampling rate:", sr)

# Save the decoded audio
sf.write("WAV/test.wav", y, sr)
print("Saved example.wav successfully.")
