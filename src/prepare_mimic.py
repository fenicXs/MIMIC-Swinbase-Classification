from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare MIMIC-CXR-JPG image-level CSVs for training (filter by view)."
    )
    p.add_argument("--mimic_root", type=str, required=True, help="Root folder containing files/ and CSVs.")
    p.add_argument("--output_dir", type=str, required=True, help="Where to write processed train/val/test CSVs.")
    p.add_argument(
        "--views",
        type=str,
        default="PA",
        help="View position to keep (PA only).",
    )
    p.add_argument(
        "--uncertainty_policy",
        type=str,
        default="u_zeros",
        choices=["u_zeros"],
        help="Uncertainty handling. Currently supports: u_zeros (-1 -> 0).",
    )
    p.add_argument(
        "--keep_unlabeled_studies",
        action="store_true",
        help="If set, keep images whose (subject_id, study_id) has no row in chexpert.csv (labels become all-zeros). "
             "By default, those rows are dropped (rare).",
    )
    return p.parse_args()


def build_dicom_to_relpath(image_filenames_path: Path) -> Dict[str, str]:
    """Build dicom_id -> rel_path map from IMAGE_FILENAMES."""
    mapping: Dict[str, str] = {}
    with open(image_filenames_path, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            dicom_id = Path(rel).stem  # filename without .jpg
            mapping[dicom_id] = rel
    return mapping


def apply_u_zeros(df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    """U-Zeros policy: 1 -> 1, else -> 0 (covers 0, -1, NaN)."""
    for c in label_cols:
        s = pd.to_numeric(df[c], errors="coerce")  # NaN for blanks
        df[c] = (s == 1.0).astype(np.float32)
    return df


def main() -> None:
    args = parse_args()

    mimic_root = Path(args.mimic_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_csv = mimic_root / "mimic-cxr-2.0.0-split.csv"
    meta_csv = mimic_root / "mimic-cxr-2.0.0-metadata.csv"
    chexpert_csv = mimic_root / "mimic-cxr-2.0.0-chexpert.csv"
    image_filenames = mimic_root / "IMAGE_FILENAMES"

    for p in [split_csv, meta_csv, chexpert_csv, image_filenames]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    if not views:
        raise ValueError("No views specified. Example: --views PA")

    print(f"[prepare] mimic_root: {mimic_root}")
    print(f"[prepare] output_dir: {out_dir}")
    print(f"[prepare] keeping views: {views}")
    print(f"[prepare] uncertainty policy: {args.uncertainty_policy}")

    # 1) dicom_id -> rel_path
    print("[prepare] Building dicom_id -> rel_path mapping from IMAGE_FILENAMES ...")
    dicom_to_path = build_dicom_to_relpath(image_filenames)
    print(f"[prepare] dicom_to_path size: {len(dicom_to_path):,}")

    # 2) Load split + minimal metadata
    print("[prepare] Loading split.csv ...")
    split_df = pd.read_csv(split_csv, usecols=["dicom_id", "study_id", "subject_id", "split"])
    print(f"[prepare] split rows: {len(split_df):,}")

    print("[prepare] Loading metadata.csv (dicom_id, ViewPosition only) ...")
    meta_df = pd.read_csv(meta_csv, usecols=["dicom_id", "ViewPosition"])
    print(f"[prepare] metadata rows: {len(meta_df):,}")

    # Merge to attach ViewPosition
    df = split_df.merge(meta_df, on="dicom_id", how="left")

    # Filter by view
    df["ViewPosition"] = df["ViewPosition"].fillna("")
    df = df[df["ViewPosition"].isin(views)].copy()
    print(f"[prepare] after view filter rows: {len(df):,}")

    # 3) Load chexpert (study-level)
    print("[prepare] Loading chexpert.csv ...")
    chexpert_df = pd.read_csv(chexpert_csv)
    if chexpert_df.shape[1] < 3:
        raise ValueError("Unexpected chexpert.csv format (need >= 3 columns).")
    label_cols = list(chexpert_df.columns[2:])  # 14 labels
    print(f"[prepare] label columns ({len(label_cols)}): {label_cols}")

    # Merge study-level labels onto image rows
    df = df.merge(
        chexpert_df,
        on=["subject_id", "study_id"],
        how="left",
        indicator=True,
    )

    if not args.keep_unlabeled_studies:
        before = len(df)
        df = df[df["_merge"] == "both"].copy()
        dropped = before - len(df)
        if dropped:
            print(f"[prepare] Dropped {dropped} images with no chexpert row (rare).")
    df = df.drop(columns=["_merge"])

    # 4) rel_path
    df["rel_path"] = df["dicom_id"].map(dicom_to_path)
    missing_paths = int(df["rel_path"].isna().sum())
    if missing_paths:
        print(f"[prepare] WARNING: {missing_paths} images missing in IMAGE_FILENAMES mapping. Dropping them.")
        df = df[df["rel_path"].notna()].copy()

    # 5) Apply uncertainty policy and finalize labels to 0/1
    if args.uncertainty_policy == "u_zeros":
        df = apply_u_zeros(df, label_cols)
    else:
        raise ValueError(f"Unsupported uncertainty_policy: {args.uncertainty_policy}")

    # Reorder columns
    keep_cols = ["rel_path", "dicom_id", "subject_id", "study_id", "split", "ViewPosition"] + label_cols
    df = df[keep_cols].copy()

    # Save class names
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump({"class_names": label_cols}, f, indent=2)

    # Save split CSVs
    for split_name in ["train", "validate", "test"]:
        split_df_out = df[df["split"] == split_name].copy()
        out_path = out_dir / f"{split_name}.csv"
        split_df_out.to_csv(out_path, index=False)
        print(f"[prepare] wrote {split_name}: {len(split_df_out):,} rows -> {out_path}")

    # Save summary
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"views={views}\n")
        for split_name in ["train", "validate", "test"]:
            n = int((df["split"] == split_name).sum())
            f.write(f"{split_name}={n}\n")
    print(f"[prepare] wrote summary -> {summary_path}")
    print("[prepare] done.")


if __name__ == "__main__":
    main()
