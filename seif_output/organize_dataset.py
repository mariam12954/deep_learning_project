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
import re
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# -- Paths --------------------------------------------------------------------
OUTPUT_DIR        = Path("seif_output")
FILTERED_CSV      = OUTPUT_DIR / "filtered_products.csv"
IMAGES_DIR        = OUTPUT_DIR / "images"

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
    text = re.sub(r"[^\w\s-]", "", str(text).lower())
    return re.sub(r"[\s_-]+", "_", text).strip("_")[:80]


def find_image(name: str, label: str) -> Path | None:
    """
    Targeted scraper saves images to: images/<label>/<slug>_1.<ext>
    Try common extensions.
    """
    slug = slugify(name)
    label_dir = IMAGES_DIR / label
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = label_dir / f"{slug}_1{ext}"
        if candidate.exists():
            return candidate
    # fallback: search by prefix
    if label_dir.exists():
        matches = list(label_dir.glob(f"{slug}_1.*"))
        if matches:
            return matches[0]
    return None


def make_split(df: pd.DataFrame):
    df_trainval, df_test = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
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
    print("\n-- Image branch --")

    df_img = df[df["has_image"] == True].copy()
    print(f"Products with has_image=True : {len(df_img)}")

    # verify file exists on disk using label-aware path
    df_img["image_path"] = df_img.apply(
        lambda r: find_image(r["name"], r["label"]), axis=1
    )
    missing = df_img["image_path"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} files marked downloaded but not found on disk - skipping")
    df_img = df_img[df_img["image_path"].notna()].copy()
    print(f"Images verified on disk      : {len(df_img)}")

    if len(df_img) < 10:
        print("ERROR: too few images. Check seif_scraper_targeted.py ran correctly.")
        return None, None, None

    # check minimum per class for stratified split
    class_counts = df_img["label"].value_counts()
    print("\nImages per class:")
    for cls, cnt in class_counts.items():
        print(f"  {cls:<22} {cnt}")

    small = class_counts[class_counts < 3].index.tolist()
    if small:
        print(f"Merging into general_medicine (less than 3 images): {small}")
        for cls in small:
            df_img.loc[df_img["label"] == cls, "label"] = "general_medicine"

    df_train, df_val, df_test = make_split(df_img)

    splits = {"train": df_train, "val": df_val, "test": df_test}
    copied  = 0
    skipped = 0

    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            src: Path = row["image_path"]
            dest_dir  = IMAGE_DATASET_DIR / split_name / row["label"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
                copied += 1
            else:
                skipped += 1

    print(f"\nCopied  : {copied}")
    print(f"Skipped : {skipped}")

    print("\nImage split summary:")
    for split_name, split_df in splits.items():
        print(f"  {split_name:<6}: {len(split_df)}")
        for cls, cnt in split_df["label"].value_counts().items():
            print(f"           {cls:<22} {cnt}")

    return df_train, df_val, df_test


# -- Text branch --------------------------------------------------------------

def build_text_branch(df: pd.DataFrame):
    print("\n-- Text branch --")
    print(f"Total rows: {len(df)}")

    TEXT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    df_train, df_val, df_test = make_split(df)

    cols = ["name", "name_clean", "text", "text_length", "label", "label_encoded"]

    df_train[cols].to_csv(TEXT_DATASET_DIR / "train.csv", index=False, encoding="utf-8-sig")
    df_val[cols].to_csv(TEXT_DATASET_DIR   / "val.csv",   index=False, encoding="utf-8-sig")
    df_test[cols].to_csv(TEXT_DATASET_DIR  / "test.csv",  index=False, encoding="utf-8-sig")

    print(f"train.csv : {len(df_train)} rows")
    print(f"val.csv   : {len(df_val)} rows")
    print(f"test.csv  : {len(df_test)} rows")

    print("\nText split summary:")
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"  {split_name:<6}: {len(split_df)}")
        for cls, cnt in split_df["label"].value_counts().items():
            print(f"           {cls:<22} {cnt}")


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  organize_dataset_v2.py")
    print("=" * 55)

    if not FILTERED_CSV.exists():
        print(f"ERROR: {FILTERED_CSV} not found.")
        print("Run filter_and_prepare_v3.py first.")
        return

    df = pd.read_csv(FILTERED_CSV, encoding="utf-8-sig")
    print(f"Loaded filtered_products.csv -> {len(df)} rows")

    df["has_image"]     = df["has_image"].astype(str).str.lower() == "true"
    df["label"]         = df["label"].astype(str)
    df["label_encoded"] = df["label_encoded"].astype(int)

    print("\nClass distribution:")
    for cls, cnt in df["label"].value_counts().items():
        print(f"  {cls:<22} {cnt}")

    build_image_branch(df)
    build_text_branch(df)

    print("\n" + "=" * 55)
    print("  Done.")
    print(f"  Image dataset -> {IMAGE_DATASET_DIR.resolve()}")
    print(f"  Text dataset  -> {TEXT_DATASET_DIR.resolve()}")
    print("  Next: python model_1_cnn.py")
    print("=" * 55)


if __name__ == "__main__":
    main()

