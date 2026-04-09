"""
================================
Reads products_data.csv + images_index.csv
and produces:
  1. filtered_products.csv
  2. dataset_report.txt
================================
"""

import pandas as pd
from pathlib import Path
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# -- Paths --------------------------------------------------------------------
OUTPUT_DIR   = Path("seif_output")
PRODUCTS_CSV = OUTPUT_DIR / "products_data.csv"
IMAGES_CSV   = OUTPUT_DIR / "images_index.csv"
FILTERED_CSV = OUTPUT_DIR / "filtered_products.csv"
REPORT_TXT   = OUTPUT_DIR / "dataset_report.txt"
# -----------------------------------------------------------------------------

# -- Class keyword rules ------------------------------------------------------
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
RARE_WORD_MIN_FREQ = 3
MAX_WORDS = 50


# ---------------------- Helper Functions -------------------------------------

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


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # for english and arabic text 
    text = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_drug_name(name):
    name = str(name).lower()
    name = re.sub(r"\d+mg|\d+\s*ml|\d+mg/ml", "", name)
    return name.strip()


# -----------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  Step 1 - Filter and Prepare Dataset")
    print("=" * 55)

    if not PRODUCTS_CSV.exists():
        print(f"\nERROR: {PRODUCTS_CSV} not found.")
        return

    df = pd.read_csv(PRODUCTS_CSV, encoding="utf-8-sig")
    print(f"\nLoaded -> {len(df)} rows")

    # initial cleaning
    
    for col in ("name", "description", "category", "subcategory"):
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").str.strip()

    # Filter medical
    df["is_medical"] = df.apply(is_medical, axis=1)
    df_med = df[df["is_medical"]].copy()

    print(f"Medical only -> {len(df_med)} rows")

    # -------------------- TEXT PREPROCESSING --------------------

    # combine text
    df_med["text"] = (
        df_med["name"] + " " +
        df_med["description"] + " " +
        df_med["category"] + " " +
        df_med["subcategory"]
    )

    df_med["text"] = df_med["text"].apply(clean_text)

    # remove duplicates
    df_med = df_med.drop_duplicates(subset=["text"])

    # remove empty
    df_med = df_med[df_med["text"].str.strip() != ""]

    # remove rare words
    tokens = df_med["text"].str.split()
    word_counts = Counter(word for row in tokens for word in row)
    common_words = {w for w, c in word_counts.items() if c >= RARE_WORD_MIN_FREQ}

    def filter_words(t):
        return " ".join([w for w in t.split() if w in common_words])

    df_med["text"] = df_med["text"].apply(filter_words)

    # limit text length
    df_med["text"] = df_med["text"].apply(
        lambda x: " ".join(x.split()[:MAX_WORDS])
    )

    # text length feature
    df_med["text_length"] = df_med["text"].apply(lambda x: len(x.split()))

    # normalize names
    df_med["name_clean"] = df_med["name"].apply(normalize_drug_name)

    # -------------------- LABELING --------------------

    df_med["label"] = df_med.apply(classify, axis=1)
    df_med["label"] = df_med["label"].fillna("general_medicine")

    # -------------------- IMAGE FILTER --------------------

    if IMAGES_CSV.exists():
        df_img = pd.read_csv(IMAGES_CSV)
        df_img = df_img[df_img["downloaded"] == True]

        img_set = set(df_img["product_name"].str.lower().str.strip())
        df_med["has_image"] = df_med["name"].str.lower().str.strip().isin(img_set)

        df_cnn = df_med[df_med["has_image"]].copy()
    else:
        df_cnn = df_med.copy()
        df_cnn["has_image"] = False

    # -------------------- BALANCE --------------------

    class_counts = df_cnn["label"].value_counts()
    small = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index

    for cls in small:
        df_cnn.loc[df_cnn["label"] == cls, "label"] = "general_medicine"

    # -------------------- ENCODING --------------------

    le = LabelEncoder()
    df_cnn["label_encoded"] = le.fit_transform(df_cnn["label"])

    # -------------------- IMAGE NAME --------------------

    df_cnn["image_name"] = (
        df_cnn["name"]
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
        + ".jpg"
    )

    # -------------------- FINAL --------------------

    df_final = df_cnn[[
        "name",
        "name_clean",
        "text",
        "text_length",
        "label",
        "label_encoded",
        "has_image",
        "image_name"
    ]]

    df_final.to_csv(FILTERED_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved -> {FILTERED_CSV}")

    # Report
    report = [
        f"Total: {len(df)}",
        f"Medical: {len(df_med)}",
        f"CNN-ready: {len(df_cnn)}",
        "",
        "Classes:"
    ]

    for cls, cnt in df_cnn["label"].value_counts().items():
        report.append(f"{cls}: {cnt}")

    REPORT_TXT.write_text("\n".join(report))

    print("\nDone ✅")


if __name__ == "__main__":
    main()