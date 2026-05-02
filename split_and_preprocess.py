"""
split_and_preprocess.py
=======================
Step 2 of the Safe Pharmacy pipeline.

Key improvement: targeted augmentation per class.
Instead of a fixed copies count, each class gets enough copies
to reach TARGET_PER_CLASS. This reduces the 10x imbalance without
discarding any images, and without making the dataset unnecessarily large.

Augmentation tiers (approximate after split):
    Healthy  (>= 100 original) -> 2-3 copies -> ~200-400 train images
    Low      (40-99  original) -> 4-5 copies -> ~160-280 train images
    Critical (<  40  original) -> 6-8 copies -> ~120-190 train images

Split first, augment after:
    Val and test always stay clean (original images only).
    This prevents data leakage and gives honest per-class accuracy.
"""

import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "dataset"

IMG_SIZE     = (224, 224)
SAVE_FORMAT  = "JPEG"
SAVE_QUALITY = 95

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

AUGMENT_TRAIN    = True
TARGET_PER_CLASS = 300   # raised from 200 — 11 classes need more copies per class
MAX_COPIES       = 10    # raised from 8 — small classes (vitamins=57) need up to 10x

IMAGE_EXTS  = {".webp", ".jpg", ".jpeg", ".png", ".jfif"}
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# AUGMENTATION
# ---------------------------------------------------------------------------

def random_augment(img: Image.Image) -> Image.Image:
    """
    Safe augmentations for drug packaging images.
    Each transform has a 50% chance of applying.
    No extreme distortion — packaging text must stay readable.
    """
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    if random.random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))

    if random.random() > 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

    if random.random() > 0.5:
        img = ImageEnhance.Saturation(img).enhance(random.uniform(0.8, 1.2))

    if random.random() > 0.5:
        w, h = img.size
        crop = random.uniform(0.82, 1.0)
        left = int(w * (1 - crop) / 2)
        top  = int(h * (1 - crop) / 2)
        img  = img.crop((left, top, w - left, h - top))
        img  = img.resize((w, h), Image.BILINEAR)

    return img


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def save_image(img: Image.Image, path: Path):
    img.save(path, format=SAVE_FORMAT, quality=SAVE_QUALITY)


def open_and_resize(src: Path) -> Image.Image:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize(IMG_SIZE, Image.LANCZOS)
        return im.copy()


def compute_aug_copies(n_train: int) -> int:
    """
    How many augmented copies per image for this class?
    Ceiling division: copies = ceil((TARGET - n) / n), capped at MAX_COPIES.
    If the class is already at or above TARGET, returns 0 (no augmentation needed).
    """
    if n_train == 0:
        return 0
    if n_train >= TARGET_PER_CLASS:
        return 0
    needed = TARGET_PER_CLASS - n_train
    copies = (needed + n_train - 1) // n_train
    return min(copies, MAX_COPIES)


# ---------------------------------------------------------------------------
# STEP 1 — SPLIT
# ---------------------------------------------------------------------------

def split_images() -> tuple[dict, dict]:
    """
    Copies images from dataset/images/<cls>/ into dataset/split/train|val|test/<cls>/.
    Splitting happens on original images only — no resizing or augmentation here.
    """
    random.seed(RANDOM_SEED)

    images_dir = Path(OUTPUT_DIR) / "images"
    split_base = Path(OUTPUT_DIR) / "split"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Folder not found: {images_dir}\n"
            "Run organize.py first to prepare the dataset."
        )

    for s in ("train", "val", "test"):
        (split_base / s).mkdir(parents=True, exist_ok=True)

    totals    = {"train": 0, "val": 0, "test": 0}
    per_class = {}

    for cls_dir in sorted(d for d in images_dir.iterdir() if d.is_dir()):
        imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        random.shuffle(imgs)

        n       = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            "train": imgs[:n_train],
            "val":   imgs[n_train: n_train + n_val],
            "test":  imgs[n_train + n_val:],
        }

        per_class[cls_dir.name] = {}
        for split_name, files in splits.items():
            dest = split_base / split_name / cls_dir.name
            dest.mkdir(exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
            totals[split_name]                  += len(files)
            per_class[cls_dir.name][split_name]  = len(files)

    return totals, per_class


def print_split_summary(totals: dict, per_class: dict):
    total_all = sum(totals.values())
    print("\n" + "=" * 68)
    print("  Split Summary  (before augmentation)")
    print("=" * 68)
    print(f"  Total  : {total_all}")
    print(f"  Train  : {totals['train']:>5}  ({totals['train']/total_all*100:.1f}%)")
    print(f"  Val    : {totals['val']:>5}  ({totals['val']/total_all*100:.1f}%)")
    print(f"  Test   : {totals['test']:>5}  ({totals['test']/total_all*100:.1f}%)")
    print("\n" + "-" * 68)
    print(f"  {'Class':<40} {'Train':>6} {'Val':>5} {'Test':>5} {'Aug copies':>10}")
    print("-" * 68)
    for cls, c in sorted(per_class.items(), key=lambda x: -x[1]["train"]):
        copies = compute_aug_copies(c["train"]) if AUGMENT_TRAIN else 0
        tag    = "no aug needed" if copies == 0 else f"+{copies}x per image"
        print(f"  {cls:<40} {c['train']:>6} {c['val']:>5} {c['test']:>5}   {tag}")


# ---------------------------------------------------------------------------
# STEP 2 — PREPROCESS + TARGETED AUGMENTATION
# ---------------------------------------------------------------------------

def preprocess_split(split_name: str, augment: bool = False) -> int:
    """
    Reads   dataset/split/<split_name>/<cls>/
    Writes  dataset/processed/<split_name>/<cls>/

    Train: resize + targeted augmentation (more copies for smaller classes).
    Val/Test: resize only — no augmentation (clean evaluation).
    """
    src_base = Path(OUTPUT_DIR) / "split"     / split_name
    dst_base = Path(OUTPUT_DIR) / "processed" / split_name

    if not src_base.exists():
        print(f"    Folder not found: {src_base}  (skipping)")
        return 0

    total_written = 0

    for cls_dir in sorted(d for d in src_base.iterdir() if d.is_dir()):
        dst_cls = dst_base / cls_dir.name
        dst_cls.mkdir(parents=True, exist_ok=True)

        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        copies = compute_aug_copies(len(images)) if augment else 0

        desc = f"  {split_name}/{cls_dir.name}"
        if copies > 0:
            desc += f" (+{copies}x aug)"

        for img_path in tqdm(images, desc=desc, leave=False):

            try:
                img = open_and_resize(img_path)
                save_image(img, dst_cls / (img_path.stem + ".jpg"))
                total_written += 1
            except Exception as e:
                print(f"\n    Skipping {img_path.name}: {e}")
                continue

            for i in range(copies):
                try:
                    aug = random_augment(open_and_resize(img_path))
                    save_image(aug, dst_cls / f"{img_path.stem}_aug{i+1}.jpg")
                    total_written += 1
                except Exception as e:
                    print(f"\n    Aug error {img_path.name}: {e}")

    print(f"   {split_name}: {total_written} total images -> {dst_base}")
    return total_written


def print_processed_summary():
    base = Path(OUTPUT_DIR) / "processed"
    print("\n" + "=" * 68)
    print("  Processed Dataset Summary  (after augmentation)")
    print("=" * 68)
    for s in ("train", "val", "test"):
        d = base / s
        if not d.exists():
            continue
        total     = sum(1 for f in d.rglob("*") if f.suffix == ".jpg")
        n_cls     = sum(1 for x in d.iterdir() if x.is_dir())
        cls_counts = sorted(
            [(x.name, sum(1 for f in x.iterdir() if f.suffix == ".jpg"))
             for x in d.iterdir() if x.is_dir()],
            key=lambda x: x[1],
        )
        if s == "train" and cls_counts:
            mn, mx = cls_counts[0][1], cls_counts[-1][1]
            ratio  = mx / mn if mn > 0 else float("inf")
            print(f"  {s:<8}: {total:>6} images | {n_cls} classes | "
                  f"min {mn} / max {mx} per class | imbalance {ratio:.1f}x")
        else:
            print(f"  {s:<8}: {total:>6} images | {n_cls} classes")
    print("=" * 68)
    print(f"\n  Processed dataset path (use in models):")
    print(f"    {base}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 68)
    print("  Safe Pharmacy — split_and_preprocess.py")
    print("=" * 68)
    print(f"\n  TARGET_PER_CLASS = {TARGET_PER_CLASS} | MAX_COPIES = {MAX_COPIES}")

    print("\n[1/4] Splitting dataset (80 / 10 / 10) ...")
    totals, per_class = split_images()
    print_split_summary(totals, per_class)

    print("\n[2/4] Preprocessing TRAIN (resize + targeted augmentation) ...")
    preprocess_split("train", augment=True)

    print("\n[3/4] Preprocessing VAL (resize only) ...")
    preprocess_split("val", augment=False)

    print("\n[4/4] Preprocessing TEST (resize only) ...")
    preprocess_split("test", augment=False)

    print_processed_summary()