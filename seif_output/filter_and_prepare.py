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
# -----------------------------------------------------------------------------

CLASS_RULES = {
    "heart": [
        "heart", "cardiac", "cardio", "warfarin", "digoxin", "nitroglycerin",
        "blood thinner", "anticoagulant", "arrhythmia", "coronary",
    ],
    "diabetes": [
        "diabetes", "diabetic", "insulin", "glucose", "metformin",
        "blood sugar", "glucometer", "hypoglycemic",
    ],
    "blood_pressure": [
        "blood pressure", "hypertension", "antihypertensive", "amlodipine",
        "losartan", "enalapril", "pressure monitor",
    ],
    "medical_device": [
        "monitor", "device", "syringe", "needle", "bandage", "glove",
        "thermometer", "nebulizer", "oximeter", "stethoscope", "catheter",
    ],
    "general_medicine": [
        "medicine", "tablet", "capsule", "syrup", "antibiotic", "vitamin",
        "supplement", "pain", "allergy", "respiratory", "pharmacy",
    ],
}

MEDICAL_FILTER = [
    "medicine", "drug", "pharma", "medical", "health", "vitamin",
    "supplement", "tablet", "capsule", "syrup", "monitor", "device",
    "syringe", "bandage", "cardiac", "diabetes", "pressure", "glucose",
    "antibiotic", "pharmacy",
]

MIN_SAMPLES_PER_CLASS = 10
RARE_WORD_MIN_FREQ    = 3
MAX_WORDS             = 50


# -- Helpers ------------------------------------------------------------------

def slugify(text: str) -> str:
    # same logic used in seif_scraper_fixed.py so image_name matches real files
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


def classify(row) -> str:
    combined = " ".join([
        str(row.get("name", "")),
        str(row.get("description", "")),
        str(row.get("category", "")),
        str(row.get("subcategory", "")),
    ]).lower()
    for class_name, keywords in CLASS_RULES.items():
        if any(kw in combined for kw in keywords):
            return class_name
    return None


def is_medical(row) -> bool:
    combined = " ".join([
        str(row.get("name", "")),
        str(row.get("category", "")),
        str(row.get("subcategory", "")),
        str(row.get("description", "")),
    ]).lower()
    return any(kw in combined for kw in MEDICAL_FILTER)


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  filter_and_prepare.py")
    print("=" * 55)

    if not PRODUCTS_CSV.exists():
        print(f"ERROR: {PRODUCTS_CSV} not found. Run seif_scraper_fixed.py first.")
        return

    df = pd.read_csv(PRODUCTS_CSV, encoding="utf-8-sig")
    print(f"Loaded products_data.csv -> {len(df)} rows")

    for col in ("name", "description", "category", "subcategory"):
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    # filter medical only
    df["is_medical"] = df.apply(is_medical, axis=1)
    df_med = df[df["is_medical"]].copy()
    print(f"Medical only             -> {len(df_med)} rows")

    # build combined text column
    df_med["text"] = (
        df_med["name"] + " " +
        df_med["description"] + " " +
        df_med["category"] + " " +
        df_med["subcategory"]
    )
    df_med["text"] = df_med["text"].apply(clean_text)

    # drop duplicates and empty rows
    df_med = df_med.drop_duplicates(subset=["text"])
    df_med = df_med[df_med["text"].str.strip() != ""]

    # remove rare words
    tokens = df_med["text"].str.split()
    word_counts = Counter(word for row in tokens for word in row)
    common_words = {w for w, c in word_counts.items() if c >= RARE_WORD_MIN_FREQ}
    df_med["text"] = df_med["text"].apply(
        lambda t: " ".join(w for w in t.split() if w in common_words)
    )

    # cap text length
    df_med["text"] = df_med["text"].apply(
        lambda t: " ".join(t.split()[:MAX_WORDS])
    )

    df_med["text_length"] = df_med["text"].apply(lambda t: len(t.split()))
    df_med["name_clean"]  = df_med["name"].apply(normalize_drug_name)

    # assign labels
    df_med["label"] = df_med.apply(classify, axis=1)
    df_med["label"] = df_med["label"].fillna("general_medicine")

    print("\nClass distribution (before image filter):")
    for cls, cnt in df_med["label"].value_counts().items():
        print(f"  {cls:<22} {cnt}")

    # -- : cast downloaded column to string before compare ----------
    if IMAGES_CSV.exists():
        df_img = pd.read_csv(IMAGES_CSV, encoding="utf-8-sig")
        df_img = df_img[df_img["downloaded"].astype(str).str.lower() == "true"]

        img_set = set(df_img["product_name"].str.lower().str.strip())
        df_med["has_image"] = df_med["name"].str.lower().str.strip().isin(img_set)

        print(f"\nWith images             : {df_med['has_image'].sum()}")
        print(f"Without images (skipped): {(~df_med['has_image']).sum()}")

        df_cnn = df_med[df_med["has_image"]].copy()
    else:
        print("WARNING: images_index.csv not found - skipping image filter")
        df_cnn = df_med.copy()
        df_cnn["has_image"] = False

    # merge small classes into general_medicine
    class_counts = df_cnn["label"].value_counts()
    small = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index.tolist()
    if small:
        print(f"\nMerging into general_medicine (too few samples): {small}")
        for cls in small:
            df_cnn.loc[df_cnn["label"] == cls, "label"] = "general_medicine"

    # encode labels
    le = LabelEncoder()
    df_cnn["label_encoded"] = le.fit_transform(df_cnn["label"])

    # -- : use slugify to match the real filenames from scraper ------
    df_cnn["image_name"] = df_cnn["name"].apply(slugify) + "_1.jpg"

    # final columns
    df_final = df_cnn[[
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

    # write report
    lines = [
        "=" * 55,
        "  Dataset Report",
        "=" * 55,
        f"Total scraped    : {len(df)}",
        f"Medical only     : {len(df_med)}",
        f"CNN-ready        : {len(df_cnn)}",
        "",
        "Final class distribution:",
    ]
    for cls, cnt in df_cnn["label"].value_counts().items():
        lines.append(f"  {cls:<22} {cnt}")
    lines += ["", "Next step: python organize_dataset.py"]

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved -> {REPORT_TXT}")
    print("\n" + "\n".join(lines))

    print("\n" + "=" * 55)
    
    print("=" * 55)


if __name__ == "__main__":
    main()
