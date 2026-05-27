"""Streamlit demo: upload a .wav, see the spectrogram and predictions."""
from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from predict import CLASS_NAMES, predict_cnn, predict_rf

st.set_page_config(page_title="Animal Sound Classifier", page_icon=":sound:")
st.title("Animal Sound Classifier")
st.caption("ESC-50 animals • Random Forest vs CNN")

MODELS_DIR = Path(__file__).parent / "models"
if not (MODELS_DIR / "rf.joblib").exists() or not (MODELS_DIR / "cnn.pt").exists():
    st.error("Models not found. Run `python train.py` first.")
    st.stop()

uploaded = st.file_uploader("Upload a .wav clip (any length — first 5s used)", type=["wav"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(tmp_path)

    y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=5)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig)
    with col2:
        mel_db = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max
        )
        fig, ax = plt.subplots(figsize=(5, 2))
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title("Mel-spectrogram")
        st.pyplot(fig)

    rf_col, cnn_col = st.columns(2)
    with rf_col:
        st.subheader("Random Forest")
        for name, prob in predict_rf(tmp_path)[:3]:
            st.write(f"**{name}** — {prob:.1%}")
            st.progress(min(float(prob), 1.0))
    with cnn_col:
        st.subheader("CNN")
        for name, prob in predict_cnn(tmp_path)[:3]:
            st.write(f"**{name}** — {prob:.1%}")
            st.progress(min(float(prob), 1.0))
else:
    st.info("Tip: grab any .wav from `data/animals/` to try the demo, or upload your own.")
    st.write("**Supported classes:**", ", ".join(CLASS_NAMES))
