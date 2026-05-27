"""Load a trained model and predict the class of a .wav file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from features import mel_spectrogram, mfcc_summary
from train import SmallCNN

PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = [
    "dog", "rooster", "pig", "cow", "frog",
    "cat", "hen", "insects", "sheep", "crow",
]


def predict_rf(wav_path: str) -> list[tuple[str, float]]:
    clf = joblib.load(PROJECT_ROOT / "models" / "rf.joblib")
    feat = mfcc_summary(wav_path).reshape(1, -1)
    probs = clf.predict_proba(feat)[0]
    return sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: -x[1])


def predict_cnn(wav_path: str) -> list[tuple[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "cnn.pt", map_location=device))
    model.eval()
    mel = mel_spectrogram(wav_path)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: -x[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="Path to a .wav file")
    parser.add_argument("--model", choices=["rf", "cnn", "both"], default="both")
    args = parser.parse_args()

    if args.model in ("rf", "both"):
        print("Random Forest top-3:")
        for name, p in predict_rf(args.wav)[:3]:
            print(f"  {name:8s} {p:.3f}")
    if args.model in ("cnn", "both"):
        print("CNN top-3:")
        for name, p in predict_cnn(args.wav)[:3]:
            print(f"  {name:8s} {p:.3f}")


if __name__ == "__main__":
    main()
