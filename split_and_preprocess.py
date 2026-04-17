
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm


# CONFIG  
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent   # فولدر المشروع نفسه
OUTPUT_DIR = BASE_DIR / "dataset"             # المشروع/dataset

#OUTPUT_DIR  = r"C:\Users\dell\Desktop\project\dataset"  # تأكد إنه نفس اللي في organize.py

IMG_SIZE    = (224, 224)   
SAVE_FORMAT = "JPEG"
SAVE_QUALITY = 95

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
#test تلقيا 0.10 (باقي النسبة)

# Augmentation – على TRAIN بس
AUGMENT_TRAIN  = True
AUGMENT_COPIES = 2      # كل صورة → +2 نسخة augmented
                        # (لو class عندها < 50 صورة ارفعه لـ 3)

IMAGE_EXTS  = {".webp", ".jpg", ".jpeg", ".png", ".jfif"}
RANDOM_SEED = 42

# AUGMENTATION


def random_augment(img: Image.Image) -> Image.Image:
    """
    Random augmentation – safe for medical/pharmacy images.
    كل operation عندها 50% احتمال تتطبق.
    """
    # Horizontal flip (أدوية symmetric غالباً)
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    # Slight rotation ±12°  (مش أكتر عشان الباكدج متلفش)
    if random.random() > 0.5:
        angle = random.uniform(-12, 12)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Brightness  ±20%
    if random.random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))

    # Contrast  ±15%
    if random.random() > 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

    # Zoom-in crop  85–100% of image
    if random.random() > 0.5:
        w, h  = img.size
        crop  = random.uniform(0.85, 1.0)
        left  = int(w * (1 - crop) / 2)
        top   = int(h * (1 - crop) / 2)
        img   = img.crop((left, top, w - left, h - top))
        img   = img.resize((w, h), Image.BILINEAR)

    return img


# HELPERS


def save_image(img: Image.Image, path: Path):
    img.save(path, format=SAVE_FORMAT, quality=SAVE_QUALITY)


def open_and_resize(src: Path) -> Image.Image:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize(IMG_SIZE, Image.LANCZOS)
        return im.copy()   # detach from file handle



# STEP 1 – SPLIT


def split_images() -> dict:
    """
    Splits dataset/images/<cls>/ → dataset/split/train|val|test/<cls>/
    Returns per_class stats dict.
    """
    random.seed(RANDOM_SEED)

    images_dir = Path(OUTPUT_DIR) / "images"
    split_base = Path(OUTPUT_DIR) / "split"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"not found: {images_dir}\n run organize.py first to prepare the images in the right structure"
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
            totals[split_name]               += len(files)
            per_class[cls_dir.name][split_name] = len(files)

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


# STEP 2 – PREPROCESS


def preprocess_split(split_name: str, augment: bool = False):
    """
    Reads  dataset/split/<split_name>/<cls>/
    Writes dataset/processed/<split_name>/<cls>/
    """
    src_base = Path(OUTPUT_DIR) / "split"     / split_name
    dst_base = Path(OUTPUT_DIR) / "processed" / split_name

    if not src_base.exists():
        print(f"    not found: {src_base}  (skip)")
        return 0

    class_dirs   = sorted(d for d in src_base.iterdir() if d.is_dir())
    total_written = 0

    for cls_dir in class_dirs:
        dst_cls = dst_base / cls_dir.name
        dst_cls.mkdir(parents=True, exist_ok=True)

        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]

        for img_path in tqdm(images, desc=f"  {split_name}/{cls_dir.name}", leave=False):
            # ---- original (resized) ----
            try:
                img = open_and_resize(img_path)
                save_image(img, dst_cls / (img_path.stem + ".jpg"))
                total_written += 1
            except Exception as e:
                print(f"\n    {img_path.name}: {e}")
                continue

            # ---- augmented copies (train only) ----
            if augment and AUGMENT_TRAIN:
                for i in range(AUGMENT_COPIES):
                    try:
                        aug = random_augment(open_and_resize(img_path))
                        save_image(aug, dst_cls / f"{img_path.stem}_aug{i+1}.jpg")
                        total_written += 1
                    except Exception as e:
                        print(f"\n    aug {img_path.name}: {e}")

    print(f"   {split_name}: {total_written} images → {dst_base}")
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
        print(f"  {s:<8}: {total:>6} images  |  {n_cls} class")

    print("=" * 60)
    print(f"\n   Processed dataset path (use this in your models):")
    print(f"     {base}")



# MAIN


if __name__ == "__main__":
    print("=" * 60)
    print("  Safe Pharmacy  split_and_preprocess.py")
    print("=" * 60)

    # ── STEP 1: Split ──────────────────────────────────────────
    print("\n Splitting dataset  (80 / 10 / 10) ...")
    totals, per_class = split_images()
    print_split_summary(totals, per_class)

    # ── STEP 2: Preprocess train (resize + augmentation) ───────
    print("\n Preprocessing TRAIN (resize + augmentation) ...")
    preprocess_split("train", augment=True)

    # ── STEP 3: Preprocess val (resize only) ───────────────────
    print("\n Preprocessing VAL (resize only) ...")
    preprocess_split("val", augment=False)

    # ── STEP 4: Preprocess test (resize only) ──────────────────
    print("\n Preprocessing TEST (resize only) ...")
    preprocess_split("test", augment=False)

    print_processed_summary()

