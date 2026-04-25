"""
organize.py
===========
Step 1 of the Safe Pharmacy pipeline.

Reads flat image folder  ->  dataset/images/<merged_class>/
- Keeps only medicine images (excludes all cosmetics, brands, baby care, etc.)
- Merges small classes (<20 images) into broader therapeutic groups
- Drops any class that remains empty after filtering
- Writes a summary report  to dataset/text/class_report.txt
- Writes class descriptions to dataset/text/class_descriptions.json
  (used by the camera inference module to display drug info at runtime)
"""

import json
import re
import shutil
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "dataset"

MIN_IMAGES_PER_CLASS = 20

IMAGE_EXTS = {".webp", ".jpg", ".jpeg", ".png", ".jfif"}

# ---------------------------------------------------------------------------
# CLASS DESCRIPTIONS
# Loaded at runtime by the camera module to display drug info.
# Each entry: { "description": "...", "note": "..." }
# ---------------------------------------------------------------------------

CLASS_INFO: dict[str, dict[str, str]] = {
    "gastrointestinal": {
        "description": "Used to treat digestive system disorders such as acidity, ulcers, nausea, and diarrhea.",
        "note": "Take as directed; some may require use before meals.",
    },
    "cns_neurology_psychiatry": {
        "description": "Used for conditions affecting the brain and nervous system such as depression, anxiety, epilepsy, and neurological disorders.",
        "note": "May cause drowsiness; do not stop abruptly without medical advice.",
    },
    "steroids_topicals": {
        "description": "Topical medications used to reduce inflammation and treat skin conditions like eczema and rashes.",
        "note": "Avoid prolonged use; apply only to affected areas.",
    },
    "cardiovascular_blood": {
        "description": "Used to manage heart conditions and blood-related disorders such as hypertension, clotting, and cholesterol.",
        "note": "Regular monitoring is important; follow dosage strictly.",
    },
    "respiratory_cough": {
        "description": "Used to relieve cough, asthma, and other respiratory tract conditions.",
        "note": "Some may cause drowsiness; avoid overdosing.",
    },
    "hormones_reproductive": {
        "description": "Used for hormonal regulation, fertility, and reproductive health conditions.",
        "note": "Use under medical supervision; may affect hormonal balance.",
    },
    "antibiotics_anti_infectives": {
        "description": "Used to treat bacterial and microbial infections.",
        "note": "Complete the full course; do not misuse or overuse.",
    },
    "diabetes_endocrine": {
        "description": "Used to control blood sugar levels and treat endocrine disorders.",
        "note": "Monitor glucose regularly; follow diet recommendations.",
    },
    "analgesics_pain_fever": {
        "description": "Used to relieve pain and reduce fever.",
        "note": "Do not exceed recommended dose; risk of liver damage if overused.",
    },
    "eye_ear_nose_preparations": {
        "description": "Used to treat infections and conditions related to eyes, ears, and nose.",
        "note": "Use hygienically; avoid contamination of applicators.",
    },
    "musculoskeletal": {
        "description": "Used for muscle pain, joint disorders, and inflammation.",
        "note": "May cause stomach irritation; take with food if needed.",
    },
    "vitamins_supplements": {
        "description": "Used to support general health and treat vitamin deficiencies.",
        "note": "Not a substitute for a balanced diet.",
    },
    "anaesthetics_speciality": {
        "description": "Used to numb pain during medical or surgical procedures.",
        "note": "Administered under professional supervision only.",
    },
    "immunology": {
        "description": "Used to support or regulate the immune system.",
        "note": "May affect immune response; follow medical advice.",
    },
    "anti_fungals": {
        "description": "Used to treat fungal infections of the skin or body.",
        "note": "Continue treatment even after symptoms improve.",
    },
    "oncology": {
        "description": "Used in the treatment of cancer and tumor-related conditions.",
        "note": "Requires strict medical supervision; may have strong side effects.",
    },
    "weight_metabolism": {
        "description": "Used to manage weight and metabolic disorders.",
        "note": "Combine with diet and exercise for best results.",
    },
    "dermatology": {
        "description": "Used to treat skin conditions such as acne, infections, and inflammation.",
        "note": "Avoid excessive use; follow application instructions.",
    },
    "general_medicine": {
        "description": "General-purpose medications used for common health conditions.",
        "note": "Use as directed; consult a doctor if symptoms persist.",
    },
}

# ---------------------------------------------------------------------------
# MERGE MAP
# ---------------------------------------------------------------------------

MERGE_MAP = {
    "analgesics_pain_fever": [
        "analgesic_a_rheumatic",
        "analgesica_rheumatic",
        "non_narcotic_analgesic",
        "headachefever",
        "migraine_treatment",
        "other_anti_rheumatics",
        "gout_treatment",
    ],
    "antibiotics_anti_infectives": [
        "anti_biotics",
        "infections",
        "topical_anti_biotic",
        "antiseptic",
        "urinary_tract_antiseptic",
        "anthelmintics",
        "anti_dysentericamoebicparas",
        "anti_virals",
        "topical_anti_viral",
        "scabies_lice",
    ],
    "steroids_topicals": [
        "steroid",
        "steroid_anti_biotic",
        "topical_steroid",
        "topical_steroid_anti_biotic",
        "anti_fungal_steroid",
        "topical_prepareation",
        "topical_anti_biotic",
        "gluco_corticoid",
        "anti_acne",
        "acne_preparations",
        "burnswounds",
        "psoriasiseczema",
        "vitiligo_treatment",
        "wartsanti_corn_preparations",
    ],
    "eye_ear_nose_preparations": [
        "eye",
        "eyeearnose",
        "eye_allergy_inflammation",
        "eye_irrigation",
        "eye_local_anaesthetic",
        "ear_preparation",
        "nose_preparation",
        "glaucoma_treatment",
        "mydriatics",
    ],
    "anti_fungals": [
        "anti_fungals",
        "anti_dandruff",
    ],
    "gastrointestinal": [
        "gastro_intestinal_tract",
        "antacids",
        "digestive",
        "anti_diarrhoeal",
        "catharticlaxativepurgative",
        "anti_emetic",
        "anti_flatulence",
        "anti_spasmodic",
        "ulcer_treatment",
        "haemorrhoids_anal_fissures",
        "liver_disease_management",
    ],
    "respiratory_cough": [
        "respiratory_system",
        "bronchodilator",
        "cough_expectorant_sedative",
        "mucolytic_muco_regulator",
        "anti_catarrhals",
        "anti_tussive",
        "lozenges",
        "topical_treatment_of_the_mouth",
    ],
    "cardiovascular_blood": [
        "cardio_vascular_system",
        "anti_hypertensives",
        "angina_treatment",
        "anti_arrhythmics",
        "anti_coagulants",
        "haemostaticscoagulants",
        "lipids_regulation",
        "congestive_heart_failure",
        "circulatory_disturbance_agent",
        "anti_hypotension",
        "vascularitics",
        "varicose_veins",
        "anaemia",
    ],
    "diabetes_endocrine": [
        "diabetes_care",
        "insulins",
        "hypo_glycaemics_antidiabetic",
        "endocrine_system",
        "anti_hyper_thyroidism",
        "anti_hypo_thyroidism",
        "growth_hormone",
        "anabolic",
    ],
    "cns_neurology_psychiatry": [
        "central_nervous_system",
        "cerebral_stimulant",
        "cns_stimulants",
        "neurotonic",
        "psychotic_disorders",
        "anti_depressant",
        "anti_epileptic_a_convulsant",
        "anti_parkinsonism",
        "alzheimer_treatment",
        "adhdattentdeficithyperacd",
        "sedatives_hypnotics",
        "nocturnal_enuresis",
        "mytonics",
    ],
    "vitamins_supplements": [
        "vitamins_or_minerals",
        "multivitamins",
        "nutrition_supplements",
        "nutrientsblood_electrolytes",
    ],
    "musculoskeletal": [
        "musculo_skeletal_system",
        "skeletal_muscle_relaxant",
        "osteoporosis_arthritis_manag",
        "osteoporosisarthritis_manag",
    ],
    "oncology": [
        "cancer_therapy",
        "alkylating_agent",
        "anti_metabolites",
        "cytostatic_anti_androgen",
        "cytostatic_anti_oestrogen",
        "cytostatic_elgonadtropin_analogu",
        "monoclonal_antibodies",
        "interferons",
        "neutropenia",
    ],
    "hormones_reproductive": [
        "female_sex_hormones",
        "male_sex_horm_androgens",
        "contraceptives",
        "infertility_treatment",
        "menopausalgyn_disorders",
        "gynaecologyurinary_tract_dis",
        "prostatic_hyperplasia",
        "anti_galactorrhoea",
        "male_sexual_tonics",
    ],
    "immunology": [
        "immunological_system",
        "immuno_suppresives",
        "immunomodulator",
        "vaccines",
    ],
    "anaesthetics_speciality": [
        "general_anaesthetic",
        "local_anaesthetic",
        "eye_local_anaesthetic",
        "anti_dote",
        "enzyme_inhibitor",
        "plasma_substituent_expander",
        "plasma_substituentexpander",
    ],
    "dermatology": [
        "anti_histaminesanti_inflam",
    ],
    "weight_metabolism": [
        "weight_control_fitness",
        "nicotine_replacement_therapy",
        "alopecia_treatment",
    ],
    "general_medicine": [
        "medicine",
    ],
}

# ---------------------------------------------------------------------------
# Build reverse lookup:  source_class  ->  target_class
# ---------------------------------------------------------------------------

SOURCE_TO_TARGET: dict[str, str] = {}
for target, sources in MERGE_MAP.items():
    for src in sources:
        SOURCE_TO_TARGET[src] = target


def extract_class(filename: str) -> str:
    """Strip extension and trailing _N index to get the raw class name."""
    name = re.sub(r"\.(webp|jpg|jpeg|png|jfif)$", "", filename, flags=re.IGNORECASE)
    return re.sub(r"_\d+$", "", name)


# ---------------------------------------------------------------------------
# STEP 1 – organize images into merged class folders
# ---------------------------------------------------------------------------

def organize_images() -> tuple[dict[str, int], int]:
    source     = Path(SOURCE_DIR)
    images_dir = Path(OUTPUT_DIR) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied:  dict[str, int] = defaultdict(int)
    skipped = 0

    for img_file in source.iterdir():
        if img_file.suffix.lower() not in IMAGE_EXTS:
            continue

        raw_class = extract_class(img_file.name)
        target    = SOURCE_TO_TARGET.get(raw_class)

        if target is None:
            skipped += 1
            continue

        dest_dir = images_dir / target
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(img_file, dest_dir / img_file.name)
        copied[target] += 1

    return dict(copied), skipped


# ---------------------------------------------------------------------------
# STEP 2 – remove classes that still fall below MIN threshold
# ---------------------------------------------------------------------------

def remove_small_classes(copied: dict[str, int]) -> dict[str, int]:
    images_dir = Path(OUTPUT_DIR) / "images"
    removed    = {}

    for cls, count in list(copied.items()):
        if count < MIN_IMAGES_PER_CLASS:
            cls_path = images_dir / cls
            if cls_path.exists():
                shutil.rmtree(cls_path)
            removed[cls] = count
            del copied[cls]

    return removed


# ---------------------------------------------------------------------------
# STEP 3 – write text report
# ---------------------------------------------------------------------------

def write_report(final: dict[str, int], removed: dict[str, int], skipped: int):
    text_dir = Path(OUTPUT_DIR) / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    # ---- class_report.txt ----
    report_path = text_dir / "class_report.txt"
    lines = []
    lines.append("=" * 60)
    lines.append("  Safe Pharmacy - Dataset Class Report")
    lines.append("=" * 60)
    lines.append(f"\nFinal classes : {len(final)}")
    lines.append(f"Total images  : {sum(final.values())}")
    lines.append(f"Non-medicine  : {skipped} images excluded")
    lines.append(f"Removed (< {MIN_IMAGES_PER_CLASS} images): {len(removed)} classes\n")
    lines.append("-" * 40)
    lines.append("  FINAL CLASSES (sorted by image count desc)")
    lines.append("-" * 40)
    for cls, count in sorted(final.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls:<45} {count:>4} images")

    if removed:
        lines.append("\n" + "-" * 40)
        lines.append("  REMOVED CLASSES (too few images after merge)")
        lines.append("-" * 40)
        for cls, count in sorted(removed.items(), key=lambda x: -x[1]):
            lines.append(f"  {cls:<45} {count:>4} images")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n  Report saved to: {report_path}")

    # ---- class_descriptions.json ----
    # Only write entries for classes that actually exist in the final dataset.
    active_info = {
        cls: CLASS_INFO[cls]
        for cls in final
        if cls in CLASS_INFO
    }
    desc_path = text_dir / "class_descriptions.json"
    desc_path.write_text(
        json.dumps(active_info, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Descriptions saved to: {desc_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Safe Pharmacy - organize.py")
    print("=" * 60)

    print("\n[1/3] Copying images to merged class folders ...")
    copied, skipped = organize_images()
    print(f"      Copied {sum(copied.values())} medicine images into {len(copied)} merged classes.")
    print(f"      Skipped {skipped} non-medicine images.")

    print(f"\n[2/3] Removing classes with fewer than {MIN_IMAGES_PER_CLASS} images ...")
    removed = remove_small_classes(copied)
    if removed:
        print(f"      Removed {len(removed)} class(es): {', '.join(removed.keys())}")
    else:
        print("      No classes removed.")

    print("\n[3/3] Writing class report and descriptions ...")
    write_report(copied, removed, skipped)

    print("\n  Done. Dataset images are in:")
    print(f"    {Path(OUTPUT_DIR) / 'images'}")
    print("\n  Class descriptions (for camera module) are in:")
    print(f"    {Path(OUTPUT_DIR) / 'text' / 'class_descriptions.json'}")