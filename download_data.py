"""Download the ESC-50 dataset and filter to the 10 animal classes.

ESC-50 ships as a single GitHub repo (~600 MB). We pull the zip, extract it,
then keep only the rows whose target index is 0-9 (the Animals fold).
"""
from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "ESC-50-master"
AUDIO_OUT = DATA_DIR / "animals"
META_OUT = DATA_DIR / "animals.csv"

ANIMAL_TARGETS = set(range(10))


def download_and_extract() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_DIR.exists():
        print(f"[skip] {RAW_DIR} already exists.")
        return
    print(f"Downloading ESC-50 from {ESC50_URL} (~600 MB, one-time)...")
    with urlopen(ESC50_URL) as resp:
        buf = io.BytesIO(resp.read())
    print("Extracting...")
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {RAW_DIR}")


def filter_animals() -> pd.DataFrame:
    meta = pd.read_csv(RAW_DIR / "meta" / "esc50.csv")
    animals = meta[meta["target"].isin(ANIMAL_TARGETS)].copy()
    AUDIO_OUT.mkdir(exist_ok=True)
    for fname in animals["filename"]:
        src = RAW_DIR / "audio" / fname
        dst = AUDIO_OUT / fname
        if not dst.exists():
            shutil.copy2(src, dst)
    animals["path"] = animals["filename"].apply(lambda f: str(AUDIO_OUT / f))
    animals.to_csv(META_OUT, index=False)
    print(f"Wrote {len(animals)} animal clips to {AUDIO_OUT}")
    print(f"Metadata: {META_OUT}")
    print("\nClass distribution:")
    print(animals["category"].value_counts().sort_index())
    return animals


if __name__ == "__main__":
    download_and_extract()
    filter_animals()
