"""Train two models on ESC-50 animals and report a side-by-side comparison.

Model A: MFCC summary features -> Random Forest (classical baseline).
Model B: Log-mel spectrograms -> small CNN (deep learning).

ESC-50 ships with a 5-fold cross-validation split in the `fold` column.
We use fold 5 as a held-out test set and folds 1-4 for training.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from features import mel_spectrogram, mfcc_summary

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_META = PROJECT_ROOT / "data" / "animals.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "figures"
TEST_FOLD = 5
SEED = 42

# -------------------- shared --------------------

def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_META)
    train = df[df["fold"] != TEST_FOLD].reset_index(drop=True)
    test = df[df["fold"] == TEST_FOLD].reset_index(drop=True)
    return train, test


# -------------------- Random Forest --------------------

def train_random_forest(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    print("\n=== Random Forest on MFCC summary ===")
    print("Extracting features...")
    X_train = np.stack([mfcc_summary(p) for p in tqdm(train["path"], desc="train")])
    X_test = np.stack([mfcc_summary(p) for p in tqdm(test["path"], desc="test")])
    y_train, y_test = train["target"].values, test["target"].values

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=SEED
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "rf.joblib")
    return {"accuracy": float(acc), "preds": preds.tolist(), "labels": y_test.tolist()}


# -------------------- CNN --------------------

class MelDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.paths = df["path"].tolist()
        self.targets = df["target"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        mel = mel_spectrogram(self.paths[idx])
        # Per-clip normalize so loudness doesn't dominate.
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return torch.from_numpy(mel).unsqueeze(0), self.targets[idx]


class SmallCNN(nn.Module):
    """A deliberately small CNN. Three conv blocks, global avg pool, linear."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.head(x)


def train_cnn(train: pd.DataFrame, test: pd.DataFrame, epochs: int = 25) -> dict:
    print("\n=== Small CNN on log-mel spectrograms ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(SEED)
    train_loader = DataLoader(MelDataset(train), batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(MelDataset(test), batch_size=16, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), torch.as_tensor(y, device=device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        print(f"  epoch {epoch:02d} loss={running / len(train_loader.dataset):.4f}")

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds.extend(model(x).argmax(1).cpu().numpy().tolist())
            labels.extend(list(y) if not torch.is_tensor(y) else y.tolist())
    acc = accuracy_score(labels, preds)
    print(f"Test accuracy: {acc:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "cnn.pt")
    return {"accuracy": float(acc), "preds": preds, "labels": labels}


# -------------------- reporting --------------------

def write_report(rf: dict, cnn: dict, class_names: list[str]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    summary = {
        "random_forest_accuracy": rf["accuracy"],
        "cnn_accuracy": cnn["accuracy"],
        "test_fold": TEST_FOLD,
        "classes": class_names,
    }
    (PROJECT_ROOT / "results.json").write_text(json.dumps(summary, indent=2))

    for name, result in [("rf", rf), ("cnn", cnn)]:
        cm = confusion_matrix(result["labels"], result["preds"])
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                    yticklabels=class_names, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name.upper()} confusion matrix (acc={result['accuracy']:.2f})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"confusion_{name}.png", dpi=120)
        plt.close()
        print(f"Wrote figures/confusion_{name}.png")

    print("\n=== Summary ===")
    print(f"Random Forest: {rf['accuracy']:.3f}")
    print(f"CNN:           {cnn['accuracy']:.3f}")
    print(f"\nFull report written to {PROJECT_ROOT / 'results.json'}")


def main() -> None:
    train, test = load_split()
    print(f"Train clips: {len(train)} | Test clips: {len(test)}")
    classes = (
        train[["target", "category"]]
        .drop_duplicates()
        .sort_values("target")["category"]
        .tolist()
    )
    rf = train_random_forest(train, test)
    cnn = train_cnn(train, test)
    write_report(rf, cnn, classes)


if __name__ == "__main__":
    main()
