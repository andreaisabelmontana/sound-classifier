"""Audio feature extraction.

Two representations:
    mfcc_summary(path) -> 1D vector of 40 numbers (mean+std of 20 MFCCs)
        Cheap, classical, feeds Random Forest.
    mel_spectrogram(path) -> 2D float32 array (128 mel bands x time frames)
        Image-like, feeds CNN.

ESC-50 clips are 5 s at 44.1 kHz. We resample to 22050 Hz to halve compute
without meaningfully hurting accuracy in this range.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import librosa

SAMPLE_RATE = 22050
CLIP_SECONDS = 5
N_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
N_MFCC = 20
N_MELS = 128
HOP_LENGTH = 512


def load_clip(path: str | Path) -> np.ndarray:
    """Load a wav, resample, and pad/truncate to a fixed length."""
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]
    return y


def mfcc_summary(path: str | Path) -> np.ndarray:
    """Mean and std of MFCCs across time -> 40-d feature vector."""
    y = load_clip(path)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def mel_spectrogram(path: str | Path) -> np.ndarray:
    """Log-mel spectrogram in dB, shape (N_MELS, time)."""
    y = load_clip(path)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
