"""Train one form-quality classifier per exercise and report honest metrics.

Every CSV in --data-dir is treated as one exercise (filename stem = exercise
name). For each exercise:

  1. stratified train/test split on the RAW collected data
  2. augmentation applied to the TRAIN split only (no leakage into the test set)
  3. StandardScaler -> RandomForest pipeline fit on the augmented train split
  4. evaluation on the untouched test split: accuracy, per-class report,
     confusion matrix image saved to --assets-dir

All models are bundled into a single payload at models/exercise_models.pkl,
with the shared feature-column list at models/feature_columns.json and a
machine-readable metrics summary at assets/metrics.json.

Usage:
    python -m formcoach.train
    python -m formcoach.train --data-dir data --test-size 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from formcoach.augment import augment_dataframe


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_exercise(
    csv_path: Path,
    test_size: float,
    copies: int,
    noise: float,
    seed: int,
    assets_dir: Path,
) -> tuple[dict, dict]:
    """Train and evaluate one exercise. Returns (model_payload, metrics)."""
    name = csv_path.stem
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Augment the train split only — augmenting before the split would leak
    # near-duplicates of test samples into training and inflate accuracy.
    train_df = X_train.copy()
    train_df.insert(0, "label", y_train.values)
    train_aug = augment_dataframe(
        train_df, copies=copies, noise_level=noise, rng=np.random.default_rng(seed)
    )
    X_train_aug = train_aug.drop(columns=["label"])
    y_train_aug = train_aug["label"]

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
    )
    pipeline.fit(X_train_aug, y_train_aug)

    class_names = pipeline.classes_.tolist()
    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    print(f"\n=== {name} ===")
    print(f"samples: {len(df)} raw ({len(X_train)} train -> {len(X_train_aug)} augmented, {len(X_test)} test)")
    print(f"test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_path = assets_dir / f"confusion_matrix_{name}.png"
    save_confusion_matrix(cm, class_names, f"{name} — form-quality confusion matrix", cm_path)
    print(f"confusion matrix -> {cm_path}")

    payload = {"model": pipeline, "class_names": class_names}
    metrics = {
        "exercise": name,
        "raw_samples": int(len(df)),
        "train_samples_augmented": int(len(X_train_aug)),
        "test_samples": int(len(X_test)),
        "test_accuracy": round(acc, 4),
        "classes": class_names,
    }
    return payload, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data", help="Directory of per-exercise CSVs (default: data)")
    parser.add_argument("--models-dir", default="models", help="Output directory for model artifacts")
    parser.add_argument("--assets-dir", default="assets", help="Output directory for confusion matrices + metrics")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction (default 0.2)")
    parser.add_argument("--copies", type=int, default=3, help="Augmented copies of the train split (default 3)")
    parser.add_argument("--noise", type=float, default=0.02, help="Augmentation noise sigma (default 0.02)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    assets_dir = Path(args.assets_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {data_dir}/")

    master_payload: dict = {}
    all_metrics: list[dict] = []
    feature_columns: list[str] | None = None

    for csv_path in csv_files:
        payload, metrics = train_exercise(
            csv_path, args.test_size, args.copies, args.noise, args.seed, assets_dir
        )
        master_payload[csv_path.stem] = payload
        all_metrics.append(metrics)
        if feature_columns is None:
            feature_columns = pd.read_csv(csv_path, nrows=0).columns.drop("label").tolist()

    model_path = models_dir / "exercise_models.pkl"
    joblib.dump(master_payload, model_path)
    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    with open(assets_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Trained {len(master_payload)} exercise model(s) -> {model_path}")
    for m in all_metrics:
        print(f"  {m['exercise']}: test accuracy {m['test_accuracy']:.2%} "
              f"({m['test_samples']} held-out samples)")
    print("=" * 50)


if __name__ == "__main__":
    main()
