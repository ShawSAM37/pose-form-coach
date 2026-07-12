#!/usr/bin/env python3
"""
Offline benchmark: 10-class exercise classifier on the Kaggle
"Physical Exercise Recognition" dataset (see benchmark/README.md for the
dataset download).

Works with datasets that provide pose features in multiple CSV files (angles,
landmarks, distances, etc.) plus a labels.csv.

Key robustness features:
- Finds CSV files case-insensitively.
- Joins all feature tables using a detected ID column (prefer 'pose_id').
- Extracts the correct label column (prefer 'pose', else common fallbacks).
- Converts features to numeric, drops unusable columns, fills NaNs.
- Safe splitting: uses stratified splits when possible, else falls back.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def _find_file_case_insensitive(folder: str, target_name: str) -> Optional[str]:
    """Return actual path to target_name inside folder ignoring case, else None."""
    folder_path = Path(folder)
    if not folder_path.exists():
        return None
    target_lower = target_name.lower()
    for p in folder_path.iterdir():
        if p.is_file() and p.name.lower() == target_lower:
            return str(p)
    return None


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _detect_id_column(df: pd.DataFrame) -> Optional[str]:
    """Prefer pose_id. Else any column that looks like an id."""
    cols = list(df.columns)
    if "pose_id" in cols:
        return "pose_id"
    # common alternatives
    for c in cols:
        cl = str(c).lower()
        if cl in ("id", "sample_id", "frame_id"):
            return c
    # heuristic: something ending with _id
    for c in cols:
        if str(c).lower().endswith("_id"):
            return c
    return None


def _detect_label_column(labels_df: pd.DataFrame) -> str:
    """
    Detect label column inside labels.csv.
    Prefer 'pose', else 'label', 'exercise', 'class', else last non-id column.
    """
    cols = list(labels_df.columns)

    # Preferred explicit names
    for candidate in ("pose", "label", "exercise", "class", "activity"):
        if candidate in cols:
            return candidate

    # If only two columns and one is an id column, use the other
    id_col = _detect_id_column(labels_df)
    if id_col and len(cols) == 2:
        other = [c for c in cols if c != id_col][0]
        return other

    # Fallback: last column
    return cols[-1]


def _to_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric; drop all-NaN columns; fill NaNs."""
    X_num = X.copy()
    for c in X_num.columns:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")

    # drop columns that are entirely NaN
    all_nan_cols = [c for c in X_num.columns if X_num[c].isna().all()]
    if all_nan_cols:
        X_num = X_num.drop(columns=all_nan_cols)

    # Fill NaNs with column median; if a column median is NaN, fill with 0
    med = X_num.median(numeric_only=True)
    X_num = X_num.fillna(med)
    X_num = X_num.fillna(0)

    return X_num


def _class_counts(y: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(y, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def _safe_train_val_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int
) -> Tuple[Tuple[pd.DataFrame, np.ndarray], Tuple[pd.DataFrame, np.ndarray], Tuple[pd.DataFrame, np.ndarray], bool]:
    """
    Try stratified splitting; if impossible (e.g., some class has <2 samples),
    fall back to non-stratified splitting so the script runs without errors.
    Returns (train, val, test, used_stratify).
    """
    from sklearn.model_selection import train_test_split

    # Decide if stratify is even theoretically possible
    counts = _class_counts(y)
    min_count = min(counts.values()) if counts else 0

    can_stratify_first = (min_count >= 2)  # required for train_test_split stratify
    used_stratify = False

    # 1) Train/Test
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if can_stratify_first else None
        )
        used_stratify = can_stratify_first
    except ValueError:
        # absolute fallback
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
        used_stratify = False

    # 2) Train/Val from temp
    # val_size is fraction of original dataset; convert to fraction of temp
    val_frac_of_temp = val_size / max(1e-9, (1.0 - test_size))

    counts_temp = _class_counts(y_temp)
    min_count_temp = min(counts_temp.values()) if counts_temp else 0
    can_stratify_second = (min_count_temp >= 2)

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_frac_of_temp,
            random_state=random_state,
            stratify=y_temp if can_stratify_second else None
        )
        used_stratify = used_stratify and can_stratify_second
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_frac_of_temp,
            random_state=random_state,
            stratify=None
        )
        used_stratify = False

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), used_stratify


# -----------------------------
# Dataset loader
# -----------------------------

@dataclass
class LoadedDataset:
    X: pd.DataFrame
    y: np.ndarray
    class_names: List[str]
    feature_columns: List[str]


class CSVLandmarkDatasetLoader:
    """
    Loads multiple feature CSVs + labels.csv and produces aligned X/y.
    """

    def __init__(self, csv_dir: str):
        self.csv_dir = csv_dir

        # default expected filenames (case-insensitive lookup)
        self.expected = {
            "angles": "angles.csv",
            "landmarks": "landmarks.csv",
            "xyz_distances": "xyz_distances.csv",
            "d3_distances": "3d_distances.csv",
            "labels": "labels.csv",
        }

    def load(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        print(f"[INFO] Loading CSVs from: {self.csv_dir}")

        for key, fname in self.expected.items():
            found = _find_file_case_insensitive(self.csv_dir, fname)
            if found is None:
                # also try common variant for 3d name
                if key == "d3_distances":
                    found = _find_file_case_insensitive(self.csv_dir, "3D_distances.csv")
            if found is None:
                print(f"  [WARN] Missing: {fname}")
                continue
            df = _read_csv(found)
            print(f"  ✓ {Path(found).name}: shape={df.shape}")
            data[key] = df

        if "labels" not in data:
            raise FileNotFoundError("labels.csv not found in --csv_dir")

        return data

    def build_feature_matrix(self, data: Dict[str, pd.DataFrame]) -> LoadedDataset:
        labels_df = data["labels"].copy()
        labels_id_col = _detect_id_column(labels_df)
        label_col = _detect_label_column(labels_df)

        if labels_id_col is None:
            # If labels.csv has no id column, fall back to row order alignment
            labels_df["_row_id"] = np.arange(len(labels_df))
            labels_id_col = "_row_id"

        labels_df = labels_df[[labels_id_col, label_col]].copy()
        labels_df = labels_df.rename(columns={label_col: "exercise_label"})

        # Start with labels as base table
        base = labels_df.drop_duplicates(subset=[labels_id_col]).set_index(labels_id_col)

        # Join each feature table on the ID column (prefer pose_id if present there too)
        feature_tables = []
        for key in ("angles", "landmarks", "xyz_distances", "d3_distances"):
            if key not in data:
                continue
            df = data[key].copy()
            id_col = _detect_id_column(df)
            if id_col is None:
                # no id => align by row order with labels
                df["_row_id"] = np.arange(len(df))
                id_col = "_row_id"
            df = df.drop_duplicates(subset=[id_col]).set_index(id_col)
            feature_tables.append(df)

        if not feature_tables:
            raise ValueError("No feature CSV files found (angles/landmarks/distances).")

        # Inner join ensures alignment only for ids present everywhere
        X_all = feature_tables[0]
        for t in feature_tables[1:]:
            X_all = X_all.join(t, how="inner", rsuffix="_dup")

        # Now join labels
        merged = X_all.join(base, how="inner")

        if merged.empty:
            raise ValueError(
                "After joining on ID, dataset is empty. "
                "Check that all CSVs share the same id column values (e.g., pose_id)."
            )

        # Separate X/y
        y_raw = merged["exercise_label"].astype(str).values
        X = merged.drop(columns=["exercise_label"])

        # Remove accidental non-feature columns (id-like duplicates, etc.)
        # If any column is exactly 'pose'/'label'/'exercise_label' etc, drop
        drop_candidates = {"pose", "label", "exercise_label"}
        X = X.drop(columns=[c for c in X.columns if str(c).lower() in drop_candidates], errors="ignore")

        # Convert to numeric and clean
        X = _to_numeric_features(X)

        # Encode labels
        class_names = sorted(pd.unique(y_raw).tolist())
        label_to_id = {name: i for i, name in enumerate(class_names)}
        y = np.array([label_to_id[v] for v in y_raw], dtype=np.int64)

        feature_columns = list(X.columns)

        # Log distribution
        counts = pd.Series(y_raw).value_counts()
        print("[INFO] Class distribution (top 20):")
        for k, v in counts.head(20).items():
            print(f"  - {k}: {int(v)}")

        return LoadedDataset(X=X, y=y, class_names=class_names, feature_columns=feature_columns)


# -----------------------------
# Model training
# -----------------------------

class TabularExerciseClassifier:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.scaler = None  # StandardScaler for SVM/MLP; harmless for trees
        self.class_names: List[str] = []
        self.feature_columns: List[str] = []

    def _make_model(self):
        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=400,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="mlogloss",
                )
            except Exception:
                print("[WARN] XGBoost not available; falling back to RandomForest.")
                self.model_type = "random_forest"

        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            )

        if self.model_type == "svm":
            from sklearn.svm import SVC
            return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)

        if self.model_type == "neural_network":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=500,
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )

        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray):
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        Xtr = self.scaler.fit_transform(X_train.values)
        Xva = self.scaler.transform(X_val.values)

        self.model = self._make_model()

        if self.model_type == "xgboost":
            # eval_set supported for XGBClassifier
            self.model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
        else:
            self.model.fit(Xtr, y_train)

        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray, class_names: List[str],
                 cm_out: str = "confusion_matrix.png"):
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        Xte = self.scaler.transform(X_test.values)
        y_pred = self.model.predict(Xte)
        acc = float(accuracy_score(y_test, y_pred))
        print(f"\n[RESULT] Test Accuracy: {acc:.4f}\n")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Confusion matrix plot (optional)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(cm_out, dpi=150)
            plt.close()
            print(f"[INFO] Saved {cm_out}")
        except Exception:
            print("[WARN] Could not plot confusion matrix (matplotlib/seaborn missing or error).")

        return acc

    def save(self, path: str):
        import joblib
        payload = {
            "model_type": self.model_type,
            "model": self.model,
            "scaler": self.scaler,
            "class_names": self.class_names,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(payload, path)
        print(f"[INFO] Saved model to: {path}")

    @staticmethod
    def load(path: str) -> "TabularExerciseClassifier":
        import joblib
        payload = joblib.load(path)
        obj = TabularExerciseClassifier(payload["model_type"])
        obj.model = payload["model"]
        obj.scaler = payload["scaler"]
        obj.class_names = payload.get("class_names", [])
        obj.feature_columns = payload.get("feature_columns", [])
        return obj

    def predict_one(self, features_row: pd.Series) -> Tuple[str, float]:
        X = features_row.values.reshape(1, -1)
        Xs = self.scaler.transform(X)
        proba = self.model.predict_proba(Xs)[0]
        idx = int(np.argmax(proba))
        return self.class_names[idx], float(proba[idx])


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
    parser.add_argument("--csv_dir", type=str, default="./csv_data")
    parser.add_argument("--model_type", choices=["xgboost", "random_forest", "svm", "neural_network"], default="xgboost")
    parser.add_argument("--model_path", type=str, default="exercise_model.pkl")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--cm_out", type=str, default="confusion_matrix.png",
                        help="Where to save the confusion matrix image")
    args = parser.parse_args()

    loader = CSVLandmarkDatasetLoader(args.csv_dir)
    data = loader.load()
    ds = loader.build_feature_matrix(data)

    # Split safely
    (X_train, y_train), (X_val, y_val), (X_test, y_test), used_stratify = _safe_train_val_test_split(
        ds.X, ds.y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )

    print(f"[INFO] Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"[INFO] Stratified split used: {used_stratify}")

    if args.mode == "train":
        clf = TabularExerciseClassifier(args.model_type)
        clf.class_names = ds.class_names
        clf.feature_columns = ds.feature_columns
        clf.fit(X_train, y_train, X_val, y_val)
        clf.evaluate(X_test, y_test, ds.class_names, cm_out=args.cm_out)

        clf.save(args.model_path)

        classes_path = os.path.splitext(args.model_path)[0] + "_classes.json"
        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump(ds.class_names, f, indent=2)
        print(f"[INFO] Saved classes to: {classes_path}")

        features_path = os.path.splitext(args.model_path)[0] + "_features.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(ds.feature_columns, f, indent=2)
        print(f"[INFO] Saved feature list to: {features_path}")

    elif args.mode == "evaluate":
        clf = TabularExerciseClassifier.load(args.model_path)
        if not clf.class_names:
            clf.class_names = ds.class_names
        clf.evaluate(X_test, y_test, clf.class_names, cm_out=args.cm_out)

    elif args.mode == "predict":
        clf = TabularExerciseClassifier.load(args.model_path)
        if not clf.class_names:
            clf.class_names = ds.class_names

        # Predict a few samples from test set
        print("\n[INFO] Sample predictions from test set:")
        n = min(10, len(X_test))
        for i in range(n):
            label, conf = clf.predict_one(X_test.iloc[i])
            print(f"  sample[{i}] -> {label} ({conf:.2%})")


if __name__ == "__main__":
    main()
