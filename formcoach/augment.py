"""Augment pose datasets with Gaussian jitter and random per-sample scaling.

Landmark coordinates are normalized, so small additive noise and mild scaling
simulate camera jitter and subject-distance variation without changing the
pose class.

Used as a library by train.py (train split only, to avoid leakage), or
standalone:

    python -m formcoach.augment --input data/squat.csv --output squat_aug.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def augment_dataframe(
    df: pd.DataFrame,
    copies: int = 3,
    noise_level: float = 0.02,
    scale_range: tuple[float, float] = (0.95, 1.05),
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return df plus `copies` jittered/scaled copies of it.

    Expects a 'label' column; all other columns are treated as numeric features.
    """
    if rng is None:
        rng = np.random.default_rng()

    labels = df["label"]
    features = df.drop(columns=["label"])

    augmented = [df]
    for _ in range(copies):
        noise = rng.normal(0.0, noise_level, features.shape)
        scales = rng.uniform(scale_range[0], scale_range[1], size=(len(features), 1))
        aug = pd.DataFrame((features + noise) * scales, columns=features.columns)
        aug.insert(0, "label", labels.values)
        augmented.append(aug)

    return pd.concat(augmented, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Input CSV (label + features)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--copies", type=int, default=3, help="Augmented copies to add (default 3)")
    parser.add_argument("--noise", type=float, default=0.02, help="Gaussian noise sigma (default 0.02)")
    parser.add_argument("--scale-min", type=float, default=0.95)
    parser.add_argument("--scale-max", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rng = np.random.default_rng(args.seed)
    out = augment_dataframe(df, args.copies, args.noise, (args.scale_min, args.scale_max), rng)
    out.to_csv(args.output, index=False)
    print(f"{args.input}: {len(df)} rows -> {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
