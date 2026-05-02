"""
organize.py  (v2 — post-diagnostic merge)
==========================================
Step 1 of the Safe Pharmacy pipeline.

CHANGES FROM v1 — based on diagnose_per_class.py results:
    19 classes  ->  11 classes

Decisions made from diagnostic output:
    DROP  (acc = 0%):
        general_medicine        0.0% acc — model cannot learn it at all
        weight_metabolism       0.0% acc — only 30 original images, too few

    MERGE (acc < 35% or overfit):
        infections_immunity     <- antibiotics + anti_fungals + immunology
        hormones_oncology       <- hormones_reproductive + oncology + anaesthetics
        neuro_musculo           <- cns_neurology_psychiatry + musculoskeletal
        respiratory_digestive   <- respiratory_cough + gastrointestinal

    KEEP (acc >= 40% or borderline with enough images):
        eye_ear_nose_preparations   84.6%
        dermatology                 66.7%  (overfit fixed by dropout in model)
        diabetes_endocrine          57.1%
        vitamins_supplements        57.1%
        steroids_topicals           45.5%
        cardiovascular_blood        40.0%
        analgesics_pain_fever       35.7%  (borderline but 300 train imgs)

Final 11 classes (expected avg ~380 train images each after merge):
    1.  eye_ear_nose_preparations
    2.  dermatology
    3.  diabetes_endocrine
    4.  vitamins_supplements
    5.  steroids_topicals
    6.  cardiovascular_blood
    7.  analgesics_pain_fever
    8.  infections_immunity
    9.  hormones_oncology
    10. neuro_musculo
    11. respiratory_digestive
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
# CLASS DESCRIPTIONS  (updated for 11 classes)
# ---------------------------------------------------------------------------

CLASS_INFO: dict[str, dict[str, str]] = {
    "eye_ear_nose_preparations": {
        "description": "Used to treat infections and conditions related to eyes, ears, and nose.",
        "note": "Use hygienically; avoid contamination of applicators.",
    },
    "dermatology": {
        "description": "Used to treat skin conditions such as acne, infections, and inflammation.",
        "note": "Avoid excessive use; follow application instructions.",
    },
    "diabetes_endocrine": {
        "description": "Used to control blood sugar levels and treat endocrine disorders.",
        "note": "Monitor glucose regularly; follow diet recommendations.",
    },
    "vitamins_supplements": {
        "description": "Used to support general health and treat vitamin or mineral deficiencies.",
        "note": "Not a substitute for a balanced diet.",
    },
    "steroids_topicals": {
        "description": "Topical medications used to reduce inflammation and treat skin conditions like eczema and rashes.",
        "note": "Avoid prolonged use; apply only to affected areas.",
    },
    "cardiovascular_blood": {
        "description": "Used to manage heart conditions and blood-related disorders such as hypertension and cholesterol.",
        "note": "Regular monitoring is important; follow dosage strictly.",
    },
    "analgesics_pain_fever": {
        "description": "Used to relieve pain and reduce fever.",
        "note": "Do not exceed recommended dose; risk of liver damage if overused.",
    },
    "infections_immunity": {
        "description": "Used to treat bacterial, fungal, and microbial infections, and to support immune function.",
        "note": "Complete the full course; do not misuse or overuse antibiotics.",
    },
    "hormones_oncology": {
        "description": "Used for hormonal regulation, reproductive health, cancer treatment, and specialty medical procedures.",
        "note": "Requires strict medical supervision; may have significant side effects.",
    },
    "neuro_musculo": {
        "description": "Used for neurological conditions, psychiatric disorders, muscle pain, and joint inflammation.",
        "note": "May cause drowsiness; do not stop abruptly without medical advice.",
    },
    "respiratory_digestive": {
        "description": "Used to treat respiratory conditions such as cough and asthma, and digestive disorders such as acidity and nausea.",
        "note": "Some may cause drowsiness; take digestive medications as directed around meals.",
    },
}

# ---------------------------------------------------------------------------
# MERGE MAP  (19 -> 11)
# DROPPED: general_medicine, weight_metabolism
# ---------------------------------------------------------------------------

MERGE_MAP: dict[str, list[str]] = {

    # ---- KEEP: eye / ear / nose ----------------------------------------
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

    # ---- KEEP: dermatology (skin — systemic/oral) -----------------------
    "dermatology": [
        "anti_histaminesanti_inflam",
        "anti_acne",
        "acne_preparations",
        "psoriasiseczema",
        "vitiligo_treatment",
        "wartsanti_corn_preparations",
    ],

    # ---- KEEP: diabetes / endocrine ------------------------------------
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

    # ---- KEEP: vitamins / supplements ----------------------------------
    "vitamins_supplements": [
        "vitamins_or_minerals",
        "multivitamins",
        "nutrition_supplements",
        "nutrientsblood_electrolytes",
    ],

    # ---- KEEP: steroids / topicals -------------------------------------
    "steroids_topicals": [
        "steroid",
        "steroid_anti_biotic",
        "topical_steroid",
        "topical_steroid_anti_biotic",
        "anti_fungal_steroid",
        "topical_prepareation",
        "topical_anti_biotic",
        "gluco_corticoid",
        "burnswounds",
    ],

    # ---- KEEP: cardiovascular / blood ----------------------------------
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

    # ---- KEEP: analgesics / pain / fever -------------------------------
    "analgesics_pain_fever": [
        "analgesic_a_rheumatic",
        "analgesica_rheumatic",
        "non_narcotic_analgesic",
        "headachefever",
        "migraine_treatment",
        "other_anti_rheumatics",
        "gout_treatment",
    ],

    # ---- MERGE: infections + antifungals + immunology ------------------
    # antibiotics 18.8%, anti_fungals 20.0%, immunology 50% overfit
    # Together they form one coherent group: fighting infection/immunity
    "infections_immunity": [
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
        "anti_fungals",
        "anti_dandruff",
        "immunological_system",
        "immuno_suppresives",
        "immunomodulator",
        "vaccines",
    ],

    # ---- MERGE: hormones + oncology + anaesthetics ---------------------
    # hormones 6.7% (nearly dropped), oncology 50% overfit,
    # anaesthetics 28.6% overfit — all specialty/supervised use
    "hormones_oncology": [
        "female_sex_hormones",
        "male_sex_horm_androgens",
        "contraceptives",
        "infertility_treatment",
        "menopausalgyn_disorders",
        "gynaecologyurinary_tract_dis",
        "prostatic_hyperplasia",
        "anti_galactorrhoea",
        "male_sexual_tonics",
        "cancer_therapy",
        "alkylating_agent",
        "anti_metabolites",
        "cytostatic_anti_androgen",
        "cytostatic_anti_oestrogen",
        "cytostatic_elgonadtropin_analogu",
        "monoclonal_antibodies",
        "interferons",
        "neutropenia",
        "general_anaesthetic",
        "local_anaesthetic",
        "anti_dote",
        "enzyme_inhibitor",
        "plasma_substituent_expander",
        "plasma_substituentexpander",
    ],

    # ---- MERGE: CNS + musculoskeletal ----------------------------------
    # cns 28.0%, musculoskeletal 28.6% — both nervous system / movement
    "neuro_musculo": [
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
        "musculo_skeletal_system",
        "skeletal_muscle_relaxant",
        "osteoporosis_arthritis_manag",
        "osteoporosisarthritis_manag",
    ],

    # ---- MERGE: respiratory + gastrointestinal -------------------------
    # respiratory 33.3%, gastrointestinal 28.0% — internal organ systems
    "respiratory_digestive": [
        "respiratory_system",
        "bronchodilator",
        "cough_expectorant_sedative",
        "mucolytic_muco_regulator",
        "anti_catarrhals",
        "anti_tussive",
        "lozenges",
        "topical_treatment_of_the_mouth",
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

    # NOTE: general_medicine and weight_metabolism are NOT listed here.
    # They are intentionally DROPPED (0% accuracy, too few/noisy images).
}

# ---------------------------------------------------------------------------
# Reverse lookup: source_class -> target_class
# ---------------------------------------------------------------------------

SOURCE_TO_TARGET: dict[str, str] = {}
for target, sources in MERGE_MAP.items():
    for src in sources:
        SOURCE_TO_TARGET[src] = target


def extract_class(filename: str) -> str:
    name = re.sub(r"\.(webp|jpg|jpeg|png|jfif)$", "", filename,
                  flags=re.IGNORECASE)
    return re.sub(r"_\d+$", "", name)


# ---------------------------------------------------------------------------
# STEP 1 — copy images into merged class folders
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
# STEP 2 — remove classes still below MIN threshold
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
# STEP 3 — write report + class_descriptions.json
# ---------------------------------------------------------------------------

def write_report(final: dict[str, int], removed: dict[str, int], skipped: int):
    text_dir = Path(OUTPUT_DIR) / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    report_path = text_dir / "class_report.txt"
    lines = []
    lines.append("=" * 60)
    lines.append("  Safe Pharmacy - Dataset Class Report  (v2: 11 classes)")
    lines.append("=" * 60)
    lines.append(f"\nFinal classes : {len(final)}")
    lines.append(f"Total images  : {sum(final.values())}")
    lines.append(f"Non-medicine  : {skipped} images excluded")
    lines.append(f"Removed (< {MIN_IMAGES_PER_CLASS} images): {len(removed)} classes\n")
    lines.append("-" * 40)
    lines.append("  FINAL CLASSES (sorted by image count desc)")
    lines.append("-" * 40)
    for cls, count in sorted(final.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls:<40} {count:>4} images")

    if removed:
        lines.append("\n" + "-" * 40)
        lines.append("  REMOVED CLASSES (too few images after merge)")
        lines.append("-" * 40)
        for cls, count in sorted(removed.items(), key=lambda x: -x[1]):
            lines.append(f"  {cls:<40} {count:>4} images")

    lines.append("\n" + "-" * 40)
    lines.append("  DROPPED BY DESIGN (0% diagnostic accuracy)")
    lines.append("-" * 40)
    lines.append("  general_medicine   — 0.0% acc, unrecognisable by model")
    lines.append("  weight_metabolism  — 0.0% acc, only 30 original images")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n  Report saved: {report_path}")

    active_info = {cls: CLASS_INFO[cls] for cls in final if cls in CLASS_INFO}
    desc_path   = text_dir / "class_descriptions.json"
    desc_path.write_text(
        json.dumps(active_info, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Descriptions saved: {desc_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Safe Pharmacy - organize.py  (v2: 19 -> 11 classes)")
    print("=" * 60)

    print("\n[1/3] Copying images to merged class folders ...")
    copied, skipped = organize_images()
    print(f"      Copied {sum(copied.values())} images into {len(copied)} classes.")
    print(f"      Skipped {skipped} non-medicine / dropped images.")

    print(f"\n[2/3] Removing classes with fewer than {MIN_IMAGES_PER_CLASS} images ...")
    removed = remove_small_classes(copied)
    if removed:
        print(f"      Removed: {', '.join(removed.keys())}")
    else:
        print("      No classes removed.")

    print("\n[3/3] Writing report and class descriptions ...")
    write_report(copied, removed, skipped)

    print("\n  Done. Next step: run split_and_preprocess.py")
    print(f"  Images in: {Path(OUTPUT_DIR) / 'images'}")