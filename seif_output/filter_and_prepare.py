"""
filter_and_prepare.py  (v2 - bugs fixed)
=========================================
Reads:
    seif_output/products_data.csv
    seif_output/images_index.csv

Produces:
    seif_output/filtered_products.csv
    seif_output/dataset_report.txt
"""

import re
import pickle
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# -- Paths --------------------------------------------------------------------
OUTPUT_DIR   = Path("seif_output")
PRODUCTS_CSV = OUTPUT_DIR / "products_data.csv"
IMAGES_CSV   = OUTPUT_DIR / "images_index.csv"
FILTERED_CSV = OUTPUT_DIR / "filtered_products.csv"
REPORT_TXT   = OUTPUT_DIR / "dataset_report.txt"
ENCODER_PKL  = OUTPUT_DIR / "label_encoder.pkl"
# -----------------------------------------------------------------------------

MIN_SAMPLES_PER_CLASS = 10
RARE_WORD_MIN_FREQ    = 2
MAX_WORDS             = 50

VALID_LABELS = {
    "heart",
    "diabetes",
    "blood_pressure",
    "medical_device",
    "general_medicine",
}


# -- Helpers ------------------------------------------------------------------

def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", str(text).lower())
    return re.sub(r"[\s_-]+", "_", text).strip("_")[:80]


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_drug_name(name: str) -> str:
    name = str(name).lower()
    name = re.sub(r"\d+\s*mg|\d+\s*ml|\d+\s*mg/ml", "", name)
    return name.strip()


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  filter_and_prepare_v3.py")
    print("=" * 55)

    if not PRODUCTS_CSV.exists():
        print(f"ERROR: {PRODUCTS_CSV} not found.")
        print("Run seif_scraper_targeted.py first.")
        return

    df = pd.read_csv(PRODUCTS_CSV, encoding="utf-8-sig")
    print(f"Loaded products_data.csv -> {len(df)} rows")

    # remove git merge conflict artifacts
    df = df[~df["name"].astype(str).str.startswith("<<<<")].copy()
    df = df[~df["name"].astype(str).str.startswith(">>>>")].copy()
    df = df[~df["name"].astype(str).str.startswith("====")].copy()
    print(f"After removing git artifacts -> {len(df)} rows")

    # clean string columns
    for col in ("name", "description", "category", "subcategory", "label"):
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    # if label column missing (old scraper output), assign general_medicine
    if "label" not in df.columns:
        df["label"] = "general_medicine"

    # keep only valid labels
    df = df[df["label"].isin(VALID_LABELS)].copy()
    print(f"Valid labels only -> {len(df)} rows")

    if len(df) == 0:
        print("ERROR: no rows left after label filter.")
        print("Check that seif_scraper_targeted.py ran correctly.")
        return

    # build combined text column
    df["text"] = (
        df["name"].fillna("") + " " +
        df.get("description", pd.Series([""] * len(df))).fillna("") + " " +
        df["category"].fillna("") + " " +
        df["subcategory"].fillna("")
    )
    df["text"] = df["text"].apply(clean_text)

    # drop duplicates and empty text
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.strip() != ""].copy()
    print(f"After dedup and empty drop -> {len(df)} rows")

    # remove rare words
    tokens = df["text"].str.split()
    word_counts = Counter(w for row in tokens for w in row)
    common_words = {w for w, c in word_counts.items() if c >= RARE_WORD_MIN_FREQ}
    df["text"] = df["text"].apply(
        lambda t: " ".join(w for w in t.split() if w in common_words)
    )

    # cap text length
    df["text"] = df["text"].apply(
        lambda t: " ".join(t.split()[:MAX_WORDS])
    )

    df["text_length"] = df["text"].apply(lambda t: len(t.split()))
    df["name_clean"]  = df["name"].apply(normalize_drug_name)

    # merge classes with too few samples into general_medicine
    class_counts = df["label"].value_counts()
    small = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index.tolist()
    if small:
        print(f"Merging into general_medicine (too few samples): {small}")
        for cls in small:
            df.loc[df["label"] == cls, "label"] = "general_medicine"

    print("\nClass distribution:")
    for cls, cnt in df["label"].value_counts().items():
        print(f"  {cls:<22} {cnt}")

    # check images
    if IMAGES_CSV.exists():
        df_img = pd.read_csv(IMAGES_CSV, encoding="utf-8-sig")
        df_img = df_img[df_img["downloaded"].astype(str).str.lower() == "true"]
        img_set = set(df_img["product_name"].str.lower().str.strip())
        df["has_image"] = df["name"].str.lower().str.strip().isin(img_set)
        print(f"\nWith images    : {df['has_image'].sum()}")
        print(f"Without images : {(~df['has_image']).sum()}")
    else:
        print("WARNING: images_index.csv not found")
        df["has_image"] = False

    # encode labels
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # image filename uses same slugify as scraper
    # targeted scraper saves to images/<label>/<slug>_1.jpg
    df["image_name"] = df.apply(
        lambda r: str(Path(r["label"]) / (slugify(r["name"]) + "_1.jpg")),
        axis=1,
    )

    # final columns
    df_final = df[[
        "name",
        "name_clean",
        "text",
        "text_length",
        "label",
        "label_encoded",
        "has_image",
        "image_name",
    ]].reset_index(drop=True)

    df_final.to_csv(FILTERED_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {FILTERED_CSV}")

    # save label encoder for use in models
    with open(ENCODER_PKL, "wb") as f:
        pickle.dump(le, f)
    print(f"Saved -> {ENCODER_PKL}")

    # write report
    lines = [
        "=" * 55,
        "  Dataset Report",
        "=" * 55,
        f"Total rows       : {len(df_final)}",
        "",
        "Class distribution:",
    ]
    for cls, cnt in df_final["label"].value_counts().items():
        lines.append(f"  {cls:<22} {cnt}")
    lines += ["", "Next step: python organize_dataset_v2.py"]

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved -> {REPORT_TXT}")
    print("\n" + "\n".join(lines))

    print("\n" + "=" * 55)
   
    print("=" * 55)


if __name__ == "__main__":
    main()
