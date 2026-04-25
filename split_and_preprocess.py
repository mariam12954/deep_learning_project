"""
split_and_preprocess.py
=======================
Step 2 of the Safe Pharmacy pipeline.

Reads  dataset/images/<class>/
->     dataset/split/train|val|test/<class>/      (raw copies, 80/10/10)
->     dataset/processed/train|val|test/<class>/  (resized 224x224 + augmentation on train)

Why this order matters
----------------------
Split first, augment after:
  - Augmented images are generated ONLY from training images.
  - Validation and test sets stay clean (no augmented versions leak in).
  - This gives an honest evaluation of real-world performance.

Why augmentation on train only
-------------------------------
  - Artificially increases training variety (rotation, flip, brightness, crop).
  - Helps the model generalise to different lighting, angles, and packaging.
  - Validation/test must reflect real unseen images, so they are only resized.

Why resize to 224x224
----------------------
  - Standard input size for MobileNetV2, VGG16, ResNet50, EfficientNet, etc.
  - Ensures all images are uniform before feeding into the models.
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
# TEST_RATIO is the remainder: 1.0 - 0.80 - 0.10 = 0.10

AUGMENT_TRAIN  = True
AUGMENT_COPIES = 2   # each training image produces 2 extra augmented copies
                     # raise to 3 for classes with fewer than 50 images

IMAGE_EXTS  = {".webp", ".jpg", ".jpeg", ".png", ".jfif"}
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# AUGMENTATION
# ---------------------------------------------------------------------------

def random_augment(img: Image.Image) -> Image.Image:
    """
    Applies a random combination of safe augmentations for drug/pharmacy images.
    Each operation has a 50% chance of being applied.
    Keeps packaging readable (no heavy distortion or colour inversion).
    """
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    if random.random() > 0.5:
        angle = random.uniform(-12, 12)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    if random.random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))

    if random.random() > 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

    if random.random() > 0.5:
        w, h = img.size
        crop = random.uniform(0.85, 1.0)
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


# ---------------------------------------------------------------------------
# STEP 1 – SPLIT
# ---------------------------------------------------------------------------

def split_images() -> tuple[dict, dict]:
    """
    Copies images from dataset/images/<cls>/ into dataset/split/train|val|test/<cls>/.
    No resizing or augmentation here — just a clean stratified split per class.
    Returns overall totals and per-class counts.
    """
    random.seed(RANDOM_SEED)

    images_dir = Path(OUTPUT_DIR) / "images"
    split_base = Path(OUTPUT_DIR) / "split"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Folder not found: {images_dir}\n"
            "Run organize.py first to prepare the dataset structure."
        )

    for s in ("train", "val", "test"):
        (split_base / s).mkdir(parents=True, exist_ok=True)

    totals    = {"train": 0, "val": 0, "test": 0}
    per_class = {}

    class_dirs = sorted(d for d in images_dir.iterdir() if d.is_dir())

    for cls_dir in class_dirs:
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
    print("\n" + "=" * 60)
    print("  Split Summary")
    print("=" * 60)
    print(f"  Total  : {total_all}")
    print(f"  Train  : {totals['train']:>5}  ({totals['train']/total_all*100:.1f}%)")
    print(f"  Val    : {totals['val']:>5}  ({totals['val']/total_all*100:.1f}%)")
    print(f"  Test   : {totals['test']:>5}  ({totals['test']/total_all*100:.1f}%)")
    print("\n" + "-" * 60)
    print(f"  {'Class':<40} {'Train':>6} {'Val':>5} {'Test':>5}")
    print("-" * 60)
    for cls, c in sorted(per_class.items()):
        print(f"  {cls:<40} {c['train']:>6} {c['val']:>5} {c['test']:>5}")


# ---------------------------------------------------------------------------
# STEP 2 – PREPROCESS
# ---------------------------------------------------------------------------

def preprocess_split(split_name: str, augment: bool = False) -> int:
    """
    Reads   dataset/split/<split_name>/<cls>/
    Writes  dataset/processed/<split_name>/<cls>/

    For train (augment=True): saves original resized + AUGMENT_COPIES augmented versions.
    For val/test (augment=False): saves only the resized original.
    """
    src_base = Path(OUTPUT_DIR) / "split"     / split_name
    dst_base = Path(OUTPUT_DIR) / "processed" / split_name

    if not src_base.exists():
        print(f"    Folder not found: {src_base}  (skipping)")
        return 0

    class_dirs    = sorted(d for d in src_base.iterdir() if d.is_dir())
    total_written = 0

    for cls_dir in class_dirs:
        dst_cls = dst_base / cls_dir.name
        dst_cls.mkdir(parents=True, exist_ok=True)

        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]

        for img_path in tqdm(images, desc=f"  {split_name}/{cls_dir.name}", leave=False):

            # Original resized copy
            try:
                img = open_and_resize(img_path)
                save_image(img, dst_cls / (img_path.stem + ".jpg"))
                total_written += 1
            except Exception as e:
                print(f"\n    Skipping {img_path.name}: {e}")
                continue

            # Augmented copies (train only)
            if augment and AUGMENT_TRAIN:
                for i in range(AUGMENT_COPIES):
                    try:
                        aug = random_augment(open_and_resize(img_path))
                        save_image(aug, dst_cls / f"{img_path.stem}_aug{i+1}.jpg")
                        total_written += 1
                    except Exception as e:
                        print(f"\n    Augmentation error {img_path.name}: {e}")

    print(f"   {split_name}: {total_written} images written -> {dst_base}")
    return total_written


def print_processed_summary():
    base = Path(OUTPUT_DIR) / "processed"
    print("\n" + "=" * 60)
    print("  Processed Dataset Summary")
    print("=" * 60)
    for s in ("train", "val", "test"):
        d = base / s
        if not d.exists():
            continue
        total = sum(1 for f in d.rglob("*") if f.suffix == ".jpg")
        n_cls = sum(1 for x in d.iterdir() if x.is_dir())
        print(f"  {s:<8}: {total:>6} images  |  {n_cls} classes")
    print("=" * 60)
    print(f"\n  Processed dataset path (use this in your models):")
    print(f"    {base}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Safe Pharmacy - split_and_preprocess.py")
    print("=" * 60)

    print("\n Splitting dataset (80 / 10 / 10) ...")
    totals, per_class = split_images()
    print_split_summary(totals, per_class)

    print("\n Preprocessing TRAIN (resize + augmentation) ...")
    preprocess_split("train", augment=True)

    print("\n Preprocessing VAL (resize only) ...")
    preprocess_split("val", augment=False)

    print("\n Preprocessing TEST (resize only) ...")
    preprocess_split("test", augment=False)

    print_processed_summary()