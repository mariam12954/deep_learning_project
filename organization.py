"""
organize.py
===========
Step 1 of the Safe Pharmacy pipeline.

Reads flat image folder  ->  dataset/images/<merged_class>/
- Keeps only medicine images (excludes all cosmetics, brands, baby care, etc.)
- Merges small classes (<20 images) into broader therapeutic groups
- Drops any class that remains empty after filtering
- Writes a summary report to dataset/text/class_report.txt
"""

import re
import shutil
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent   # فولدر المشروع نفسه
SOURCE_DIR = BASE_DIR / "images"               # المشروع/images
OUTPUT_DIR = BASE_DIR / "dataset"             # المشروع/dataset
#SOURCE_DIR = r"C:\Users\dell\Desktop\project\images"
#OUTPUT_DIR = r"C:\Users\dell\Desktop\project\dataset"

MIN_IMAGES_PER_CLASS = 20        # classes below this threshold get merged

IMAGE_EXTS = {".webp", ".jpg", ".jpeg", ".png", ".jfif"}

# ---------------------------------------------------------------------------
# MERGE MAP
# Each key is the TARGET class name.
# Each value is a list of SOURCE class names that will be merged into it.
# Classes not listed here that pass the MIN threshold keep their own folder.
# ---------------------------------------------------------------------------

MERGE_MAP = {

    # ---- ANALGESICS / PAIN / FEVER / RHEUMATIC ---------------------------
    "analgesics_pain_fever": [
        "analgesic_a_rheumatic",
        "analgesica_rheumatic",
        "non_narcotic_analgesic",
        "headachefever",
        "migraine_treatment",
        "other_anti_rheumatics",
        "gout_treatment",
    ],

    # ---- ANTIBIOTICS / ANTI-INFECTIVES -----------------------------------
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

    # ---- STEROIDS / ANTI-INFLAMMATORY TOPICALS ---------------------------
    "steroids_topicals": [
        "steroid",
        "steroid_anti_biotic",
        "topical_steroid",
        "topical_steroid_anti_biotic",
        "anti_fungal_steroid",
        "topical_prepareation",
        "topical_anti_biotic",           # also antibiotics but mostly topical overlap
        "gluco_corticoid",
        "anti_acne",
        "acne_preparations",
        "burnswounds",
        "psoriasiseczema",
        "vitiligo_treatment",
        "wartsanti_corn_preparations",
    ],

    # ---- EYE / EAR / NOSE PREPARATIONS ----------------------------------
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

    # ---- ANTI-FUNGALS ----------------------------------------------------
    "anti_fungals": [
        "anti_fungals",
        "anti_dandruff",
    ],

    # ---- GASTROINTESTINAL -----------------------------------------------
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

    # ---- RESPIRATORY / COUGH / BRONCHO ----------------------------------
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

    # ---- CARDIOVASCULAR / BLOOD -----------------------------------------
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

    # ---- DIABETES / ENDOCRINE -------------------------------------------
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

    # ---- CNS / NEUROLOGY / PSYCHIATRY -----------------------------------
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

    # ---- VITAMINS / SUPPLEMENTS -----------------------------------------
    "vitamins_supplements": [
        "vitamins_or_minerals",
        "multivitamins",
        "nutrition_supplements",
        "nutrientsblood_electrolytes",
    ],

    # ---- MUSCULOSKELETAL ------------------------------------------------
    "musculoskeletal": [
        "musculo_skeletal_system",
        "skeletal_muscle_relaxant",
        "osteoporosis_arthritis_manag",
        "osteoporosisarthritis_manag",
    ],

    # ---- ONCOLOGY -------------------------------------------------------
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

    # ---- HORMONES / REPRODUCTIVE ----------------------------------------
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

    # ---- IMMUNOLOGY -----------------------------------------------------
    "immunology": [
        "immunological_system",
        "immuno_suppresives",
        "immunomodulator",
        "vaccines",
    ],

    # ---- ANAESTHETICS / SPECIALITY --------------------------------------
    "anaesthetics_speciality": [
        "general_anaesthetic",
        "local_anaesthetic",
        "eye_local_anaesthetic",
        "anti_dote",
        "enzyme_inhibitor",
        "plasma_substituent_expander",
        "plasma_substituentexpander",
    ],

    # ---- DERMATOLOGY (SYSTEMIC / ORAL) ----------------------------------
    "dermatology": [
        "anti_histaminesanti_inflam",
    ],

    # ---- WEIGHT / METABOLISM --------------------------------------------
    "weight_metabolism": [
        "weight_control_fitness",
        "nicotine_replacement_therapy",
        "alopecia_treatment",
    ],

    # ---- GENERAL MEDICINE -----------------------------------------------
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

def organize_images() -> dict[str, int]:
    source    = Path(SOURCE_DIR)
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
            # Class is not medicine or not mapped -> skip
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
    report_path = text_dir / "class_report.txt"

    lines = []
    lines.append("=" * 60)
    lines.append("  Safe Pharmacy – Dataset Class Report")
    lines.append("=" * 60)
    lines.append(f"\nFinal classes  : {len(final)}")
    lines.append(f"Total images   : {sum(final.values())}")
    lines.append(f"Non-medicine   : {skipped} images excluded")
    lines.append(f"Removed (< {MIN_IMAGES_PER_CLASS} images): {len(removed)} classes\n")

    lines.append("-" * 40)
    lines.append("  FINAL CLASSES  (sorted by image count desc)")
    lines.append("-" * 40)
    for cls, count in sorted(final.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls:<45} {count:>4} images")

    if removed:
        lines.append("\n" + "-" * 40)
        lines.append("  REMOVED CLASSES  (too few images after merge)")
        lines.append("-" * 40)
        for cls, count in sorted(removed.items(), key=lambda x: -x[1]):
            lines.append(f"  {cls:<45} {count:>4} images")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n  Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Safe Pharmacy – organize.py")
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

    print("\n[3/3] Writing class report ...")
    write_report(copied, removed, skipped)

    print("\n  Done. Dataset images are in:")
    print(f"    {Path(OUTPUT_DIR) / 'images'}")