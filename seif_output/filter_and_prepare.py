"""

================================
Reads products_data.csv + images_index.csv
and produces:
  1. filtered_products.csv  - medical products only, with class labels
  2. dataset_report.txt     - summary of counts per class

Output:
  seif_output/
  +-- filtered_products.csv
  +-- dataset_report.txt
"""

import pandas as pd
from pathlib import Path

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
# -----------------------------------------------------------------------------

MIN_SAMPLES_PER_CLASS = 10


def classify(row) -> str:
    combined = " ".join([
        str(row.get("name", "")),
        str(row.get("description", "")),
        str(row.get("category", "")),
        str(row.get("subcategory", "")),
    ]).lower()

    for class_name, keywords in CLASS_RULES.items():
        if any(kw.lower() in combined for kw in keywords):
            return class_name
    return None


def is_medical(row) -> bool:
    combined = " ".join([
        str(row.get("name", "")),
        str(row.get("category", "")),
        str(row.get("subcategory", "")),
        str(row.get("description", "")),
    ]).lower()
    return any(kw.lower() in combined for kw in MEDICAL_FILTER)


def main():
    print("=" * 55)
    print("  Step 1 - Filter and Prepare Dataset")
    print("=" * 55)

    if not PRODUCTS_CSV.exists():
        print(f"\nERROR: {PRODUCTS_CSV} not found.")
        print("Run seif_scraper_fixed.py first.")
        return

    df = pd.read_csv(PRODUCTS_CSV, encoding="utf-8-sig")
    print(f"\nLoaded products_data.csv -> {len(df)} rows")

    # Step A: keep medical products only
    df["is_medical"] = df.apply(is_medical, axis=1)
    df_med = df[df["is_medical"]].copy()
    print(f"Medical products only   -> {len(df_med)} rows")

    # Step B: assign class label
    df_med["label"] = df_med.apply(classify, axis=1)
    df_med["label"] = df_med["label"].fillna("general_medicine")

    print("\nClass distribution (before image filter):")
    for cls, count in df_med["label"].value_counts().items():
        print(f"  {cls:<22} {count}")

    # Step C: keep only products that have downloaded images
    if IMAGES_CSV.exists():
        df_img = pd.read_csv(IMAGES_CSV, encoding="utf-8-sig")
        df_img = df_img[df_img["downloaded"] == True]

        products_with_imgs = set(df_img["product_name"].str.lower().str.strip())
        df_med["has_image"] = df_med["name"].str.lower().str.strip().isin(products_with_imgs)

        print(f"\nWith downloaded images  : {df_med['has_image'].sum()}")
        print(f"Without images (skipped): {(~df_med['has_image']).sum()}")

        df_cnn = df_med[df_med["has_image"]].copy()
    else:
        print("\nWARNING: images_index.csv not found - skipping image filter")
        df_cnn = df_med.copy()
        df_cnn["has_image"] = False

    # Step D: merge classes with too few samples into general_medicine
    print("\nSamples per class (CNN-ready):")
    class_counts = df_cnn["label"].value_counts()
    small_classes = []
    for cls, count in class_counts.items():
        flag = "" if count >= MIN_SAMPLES_PER_CLASS else "  <-- too few, will merge"
        print(f"  {cls:<22} {count}{flag}")
        if count < MIN_SAMPLES_PER_CLASS:
            small_classes.append(cls)

    if small_classes:
        print(f"\nMerging into general_medicine: {small_classes}")
        for cls in small_classes:
            df_cnn.loc[df_cnn["label"] == cls, "label"] = "general_medicine"

    # Step E: save filtered CSV
    df_cnn.to_csv(FILTERED_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {FILTERED_CSV}")

    # Step F: write report
    lines = [
        "=" * 55,
        "  Dataset Report - Seif Medical Scraper",
        "=" * 55,
        f"Total scraped products   : {len(df)}",
        f"Medical products         : {len(df_med)}",
        f"CNN-ready (have images)  : {len(df_cnn)}",
        "",
        "Final class distribution:",
    ]
    for cls, count in df_cnn["label"].value_counts().items():
        lines.append(f"  {cls:<22} {count}")

    lines += ["", "Raw categories found in scrape:"]
    for cat in df["category"].dropna().unique():
        lines.append(f"  - {cat}")


    report_text = "\n".join(lines)
    REPORT_TXT.write_text(report_text, encoding="utf-8")
    print(f"Saved -> {REPORT_TXT}")
    print("\n" + report_text)

    print("\n" + "=" * 55)
 
    print("=" * 55)


if __name__ == "__main__":
    main()