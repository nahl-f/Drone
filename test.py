import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os

st.set_page_config(page_title="Drone Audio Visualizer", layout="wide")
st.title("🎧 Drone Audio Visualizer")

# --- Absolute path to your audio file ---
AUDIO_PATH = r"C:\Users\nahlf\Desktop\Drone\WAV\test.wav"

# --- Check file exists ---
if not os.path.exists(AUDIO_PATH):
    st.error(f"Audio file not found at:\n{AUDIO_PATH}")
    st.stop()

# --- Load the audio ---
try:
    y, sr = sf.read(AUDIO_PATH)
    if y.ndim > 1:
        y = y[:, 0]  # convert stereo to mono
except Exception:
    y, sr = librosa.load(AUDIO_PATH, sr=None)

# --- Audio player ---
st.subheader("Playback")
st.audio(AUDIO_PATH, format="audio/wav")

col1, col2 = st.columns(2)
# Waveform plot
with col1:
    st.subheader("Waveform")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='steelblue')
    ax1.set(xlabel='Time (s)', ylabel='Amplitude')
    ax1.set_title('Waveform', fontsize=10)
    st.pyplot(fig1, use_container_width=True)

# Mel Spectrogram plot
with col2:
    st.subheader("Mel Spectrogram")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title('Mel Spectrogram', fontsize=10)
    st.pyplot(fig2, use_container_width=True)
