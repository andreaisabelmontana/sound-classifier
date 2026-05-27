"""Exploration notebook (cell-marked).

Open this in VS Code or Jupyter — the `# %%` markers turn each block into a
runnable cell. Walks through audio fundamentals before any model code.

Run order:
    python download_data.py     # one-time, ~600 MB
    open this file as a notebook
"""

# %% [markdown]
# # ESC-50 Animals: from waveform to prediction
#
# Goals:
# 1. See what audio data actually looks like.
# 2. Turn sound into something a model can read (spectrograms, MFCCs).
# 3. Train a Random Forest baseline.
# 4. Train a small CNN on spectrograms.
# 5. Compare honestly on a held-out fold.

# %%
from pathlib import Path

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_META = Path("data/animals.csv")
df = pd.read_csv(DATA_META)
print(f"{len(df)} clips across {df['category'].nunique()} classes")
df.head()

# %% [markdown]
# ## 1. Listen and look at one clip
# A 5-second dog bark, raw amplitude over time.

# %%
sample = df[df["category"] == "dog"].iloc[0]
y, sr = librosa.load(sample["path"], sr=22050, mono=True)

fig, ax = plt.subplots(figsize=(10, 2.5))
librosa.display.waveshow(y, sr=sr, ax=ax)
ax.set_title(f"{sample['category']} — waveform")
plt.show()

ipd.Audio(sample["path"])

# %% [markdown]
# ## 2. Same clip as a mel-spectrogram
# Spectrograms turn audio into a 2D image: time on x, frequency on y, intensity as color.
# This is what the CNN will see.

# %%
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

fig, ax = plt.subplots(figsize=(10, 4))
img = librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", sr=sr, ax=ax)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set_title(f"{sample['category']} — log-mel spectrogram")
plt.show()

# %% [markdown]
# ## 3. One clip per class (sanity check)

# %%
fig, axes = plt.subplots(2, 5, figsize=(16, 6))
for ax, (_, row) in zip(axes.flat, df.groupby("category").head(1).iterrows()):
    y_c, _ = librosa.load(row["path"], sr=22050, mono=True, duration=5)
    mel_c = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y_c, sr=22050, n_mels=64), ref=np.max
    )
    librosa.display.specshow(mel_c, sr=22050, ax=ax)
    ax.set_title(row["category"])
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Extract MFCC features and train Random Forest
# MFCCs (mel-frequency cepstral coefficients) compress a spectrogram into a
# small set of coefficients per frame. Taking the mean and std across time
# gives a fixed-size vector per clip — perfect for a tree model.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from features import mfcc_summary

train_df = df[df["fold"] != 5]
test_df = df[df["fold"] == 5]

X_train = np.stack([mfcc_summary(p) for p in train_df["path"]])
X_test = np.stack([mfcc_summary(p) for p in test_df["path"]])
y_train = train_df["target"].values
y_test = test_df["target"].values

rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(f"RF accuracy: {accuracy_score(y_test, rf_preds):.3f}")
print(classification_report(y_test, rf_preds, target_names=sorted(df["category"].unique())))

# %% [markdown]
# ## 5. Train the CNN
# Run `python train.py` from the project root — it trains both models, saves
# them under `models/`, and writes confusion matrices to `figures/`.

# %% [markdown]
# ## 6. Inspect the most confused classes
# Which animals does the model mix up? Often: cow vs sheep, hen vs rooster.
# Listen to a few misclassifications and form your own hypothesis.

# %%
wrong = test_df.reset_index(drop=True).loc[rf_preds != y_test].copy()
wrong["predicted"] = [df[df["target"] == t]["category"].iloc[0] for t in rf_preds[rf_preds != y_test]]
wrong[["filename", "category", "predicted"]].head(10)
