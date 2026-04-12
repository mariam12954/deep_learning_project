"""
organize_dataset.py
====================
Reads:
    seif_output/filtered_products.csv
    seif_output/images/               (downloaded images from scraper)

Produces:
    dataset/
    +-- image/
    |   +-- train/
    |   |   +-- heart/
    |   |   +-- diabetes/
    |   |   +-- blood_pressure/
    |   |   +-- medical_device/
    |   |   +-- general_medicine/
    |   +-- val/
    |   |   +-- (same structure)
    |   +-- test/
    |       +-- (same structure)
    |
    +-- text/
        +-- train.csv
        +-- val.csv
        +-- test.csv

Split ratio: 70% train / 15% val / 15% test  (stratified by label)
"""

import shutil
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# -- Paths --------------------------------------------------------------------
OUTPUT_DIR   = Path("seif_output")
FILTERED_CSV = OUTPUT_DIR / "filtered_products.csv"
IMAGES_DIR   = OUTPUT_DIR / "images"

DATASET_DIR       = Path("dataset")
IMAGE_DATASET_DIR = DATASET_DIR / "image"
TEXT_DATASET_DIR  = DATASET_DIR / "text"
# -----------------------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


# -- Helpers ------------------------------------------------------------------

def slugify(text: str) -> str:
    # must match the same slugify used in seif_scraper_fixed.py
    text = re.sub(r"[^\w\s-]", "", str(text).lower())
    return re.sub(r"[\s_-]+", "_", text).strip("_")[:80]


def find_image(name: str, images_dir: Path) -> Path | None:
    """
    Try to find the downloaded image for a product.
    The scraper saves images as:  images/<slug>_1.jpg  (or .png etc.)
    """
    slug = slugify(name)
    # check common extensions
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images_dir / f"{slug}_1{ext}"
        if candidate.exists():
            return candidate
    # fallback: search by prefix in case extension differs
    matches = list(images_dir.glob(f"{slug}_1.*"))
    return matches[0] if matches else None


def make_split(df: pd.DataFrame):
    """
    Stratified split into train / val / test.
    Returns three DataFrames.
    """
    # first split off test
    df_trainval, df_test = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    # then split train/val from the remainder
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED,
        stratify=df_trainval["label"],
    )
    return df_train, df_val, df_test


# -- Image branch -------------------------------------------------------------

def build_image_branch(df: pd.DataFrame):
    """
    Copies images into dataset/image/train|val|test/<label>/ folders.
    Skips products where the image file is not found on disk.
    """
    print("\n-- Image branch --")

    df_img = df[df["has_image"] == True].copy()
    print(f"Products with images: {len(df_img)}")

    # verify file actually exists on disk
    df_img["image_path"] = df_img["name"].apply(
        lambda n: find_image(n, IMAGES_DIR)
    )
    missing = df_img["image_path"].isna().sum()
    if missing:
        print(f"WARNING: {missing} products marked has_image=True but file not found on disk - skipping them")
    df_img = df_img[df_img["image_path"].notna()].copy()
    print(f"Images found on disk: {len(df_img)}")

    if len(df_img) < 10:
        print("ERROR: too few images found. Check that seif_scraper_fixed.py ran and downloaded images.")
        return

    df_train, df_val, df_test = make_split(df_img)

    splits = {"train": df_train, "val": df_val, "test": df_test}
    copied_total = 0
    skipped_total = 0

    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            src: Path = row["image_path"]
            dest_dir  = IMAGE_DATASET_DIR / split_name / row["label"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
                copied_total += 1
            else:
                skipped_total += 1

    print(f"Copied : {copied_total} images")
    print(f"Skipped (already exist): {skipped_total}")

    # print split counts per class
    print("\nImage split summary:")
    for split_name, split_df in splits.items():
        print(f"  {split_name:<6}: {len(split_df)} total")
        for cls, cnt in split_df["label"].value_counts().items():
            print(f"           {cls:<22} {cnt}")

    return df_train, df_val, df_test


# -- Text branch --------------------------------------------------------------

def build_text_branch(df: pd.DataFrame):
    """
    Splits ALL medical products (with or without images) into
    dataset/text/train.csv, val.csv, test.csv.

    The text model uses 'text' as input and 'label_encoded' as target.
    It is used for products with no image or as a fallback when CNN
    confidence is low.
    """
    print("\n-- Text branch --")
    print(f"Total rows for text model: {len(df)}")

    TEXT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    df_train, df_val, df_test = make_split(df)

    cols = ["name", "name_clean", "text", "text_length", "label", "label_encoded"]

    df_train[cols].to_csv(TEXT_DATASET_DIR / "train.csv", index=False, encoding="utf-8-sig")
    df_val[cols].to_csv(TEXT_DATASET_DIR  / "val.csv",   index=False, encoding="utf-8-sig")
    df_test[cols].to_csv(TEXT_DATASET_DIR / "test.csv",  index=False, encoding="utf-8-sig")

    print(f"train.csv : {len(df_train)} rows")
    print(f"val.csv   : {len(df_val)} rows")
    print(f"test.csv  : {len(df_test)} rows")

    print("\nText split summary:")
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"  {split_name:<6}: {len(split_df)} total")
        for cls, cnt in split_df["label"].value_counts().items():
            print(f"           {cls:<22} {cnt}")


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  organize_dataset.py")
    print("=" * 55)

    if not FILTERED_CSV.exists():
        print(f"ERROR: {FILTERED_CSV} not found. Run filter_and_prepare_v2.py first.")
        return

    df = pd.read_csv(FILTERED_CSV, encoding="utf-8-sig")
    print(f"Loaded filtered_products.csv -> {len(df)} rows")

    # cast types
    df["has_image"]     = df["has_image"].astype(str).str.lower() == "true"
    df["label"]         = df["label"].astype(str)
    df["label_encoded"] = df["label_encoded"].astype(int)

    print("\nClass distribution in filtered_products.csv:")
    for cls, cnt in df["label"].value_counts().items():
        print(f"  {cls:<22} {cnt}")

    # -- Image branch
    build_image_branch(df)

    # -- Text branch (all rows, no image requirement)
    build_text_branch(df)

    print("\n" + "=" * 55)
    print("  Done.")
    print(f"  Image dataset -> {IMAGE_DATASET_DIR.resolve()}")
    print(f"  Text dataset  -> {TEXT_DATASET_DIR.resolve()}")



if __name__ == "__main__":
    main()
