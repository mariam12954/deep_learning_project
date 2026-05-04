"""
split_and_preprocess_v3.py  —  Safe Pharmacy
=============================================
Key changes from v2:
  • Split: 70 / 20 / 10  (was 80/10/10)
    → Bigger val set → overfitting caught much earlier
    → EarlyStopping on val_loss is now more reliable

  • TARGET_PER_CLASS = 350  (slight raise for small classes)
  • MAX_COPIES      = 12    (vitamins=57 needs ~12x to reach 350 in train)

  • Augmentation is more conservative:
    - Max rotation ±12° (was ±15°) — packaging text stays readable
    - No shear (removed) — distorts pill shapes
    - Brightness range 0.80–1.20 (same)
    - Added: random erasing 10% chance — forces model to not rely on
      one patch, reduces overfitting

  • Val and Test: resize only, ZERO augmentation (clean evaluation)
"""

import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
from tqdm import tqdm

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "dataset"

IMG_SIZE     = (224, 224)
SAVE_FORMAT  = "JPEG"
SAVE_QUALITY = 95

# ── NEW SPLIT ──────────────────────────────────────────────────
TRAIN_RATIO = 0.70   # was 0.80
VAL_RATIO   = 0.20   # was 0.10
# TEST = remainder (~0.10)

# ── Augmentation targets ───────────────────────────────────────
AUGMENT_TRAIN    = True
TARGET_PER_CLASS = 350   # aim for ~350 train images per class
MAX_COPIES       = 12    # vitamins (57 imgs → 40 train) needs ~9x

IMAGE_EXTS  = {".webp", ".jpg", ".jpeg", ".png", ".jfif"}
RANDOM_SEED = 42


# ── Augmentation (conservative, anti-overfit) ──────────────────

def random_augment(img: Image.Image) -> Image.Image:
    """
    Conservative augmentation for drug packaging.
    Goal: diversity without destroying visual features.
    """
    # Horizontal flip (50%)
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    # Small rotation ±12° only (was ±15°)
    if random.random() > 0.5:
        img = img.rotate(random.uniform(-12, 12),
                         resample=Image.BILINEAR, expand=False)

    # Brightness (80%)
    if random.random() > 0.2:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.80, 1.20))

    # Contrast (60%)
    if random.random() > 0.4:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

    # Saturation (50%)
    if random.random() > 0.5:
        img = ImageEnhance.Saturation(img).enhance(random.uniform(0.85, 1.15))

    # Random crop & resize (50%) — mild zoom 88–100%
    if random.random() > 0.5:
        w, h  = img.size
        crop  = random.uniform(0.88, 1.0)
        left  = int(w * (1 - crop) / 2)
        top   = int(h * (1 - crop) / 2)
        img   = img.crop((left, top, w - left, h - top))
        img   = img.resize((w, h), Image.BILINEAR)

    # Random erasing 10% — small black patch to reduce overfit
    if random.random() > 0.90:
        w, h    = img.size
        rw, rh  = int(w * random.uniform(0.05, 0.15)), int(h * random.uniform(0.05, 0.15))
        rx      = random.randint(0, w - rw)
        ry      = random.randint(0, h - rh)
        draw    = ImageDraw.Draw(img)
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=(0, 0, 0))

    return img


def save_image(img: Image.Image, path: Path):
    img.save(path, format=SAVE_FORMAT, quality=SAVE_QUALITY)


def open_and_resize(src: Path) -> Image.Image:
    with Image.open(src) as im:
        return im.convert("RGB").resize(IMG_SIZE, Image.LANCZOS).copy()


def compute_aug_copies(n_train: int) -> int:
    if n_train == 0 or n_train >= TARGET_PER_CLASS:
        return 0
    needed = TARGET_PER_CLASS - n_train
    copies = (needed + n_train - 1) // n_train
    return min(copies, MAX_COPIES)


# ── Step 1: Split ──────────────────────────────────────────────

def split_images():
    random.seed(RANDOM_SEED)
    images_dir = Path(OUTPUT_DIR) / "images"
    split_base = Path(OUTPUT_DIR) / "split"

    if not images_dir.exists():
        raise FileNotFoundError(f"Not found: {images_dir}\nRun organize_v3.py first.")

    for s in ("train", "val", "test"):
        (split_base / s).mkdir(parents=True, exist_ok=True)

    totals, per_class = {"train": 0, "val": 0, "test": 0}, {}

    for cls_dir in sorted(d for d in images_dir.iterdir() if d.is_dir()):
        imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        random.shuffle(imgs)
        n       = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        splits  = {
            "train": imgs[:n_train],
            "val":   imgs[n_train: n_train + n_val],
            "test":  imgs[n_train + n_val:],
        }
        per_class[cls_dir.name] = {}
        for sname, files in splits.items():
            dest = split_base / sname / cls_dir.name
            dest.mkdir(exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
            totals[sname]                    += len(files)
            per_class[cls_dir.name][sname]    = len(files)

    return totals, per_class


def print_split_summary(totals, per_class):
    total_all = sum(totals.values())
    print("\n" + "=" * 70)
    print("  Split Summary (70 / 20 / 10)")
    print("=" * 70)
    print(f"  Total  : {total_all}")
    for s in ("train", "val", "test"):
        pct = totals[s] / total_all * 100
        print(f"  {s:<6} : {totals[s]:>5}  ({pct:.1f}%)")
    print("\n" + "-" * 70)
    print(f"  {'Class':<40} {'Train':>6} {'Val':>5} {'Test':>5} {'Aug':>10}")
    print("-" * 70)
    for cls, c in sorted(per_class.items(), key=lambda x: -x[1]["train"]):
        copies = compute_aug_copies(c["train"]) if AUGMENT_TRAIN else 0
        tag    = "—" if copies == 0 else f"+{copies}x"
        print(f"  {cls:<40} {c['train']:>6} {c['val']:>5} {c['test']:>5}   {tag}")
    print("-" * 70)


# ── Step 2: Preprocess + Augment ──────────────────────────────

def preprocess_split(split_name: str, augment: bool = False) -> int:
    src_base = Path(OUTPUT_DIR) / "split"     / split_name
    dst_base = Path(OUTPUT_DIR) / "processed" / split_name
    if not src_base.exists():
        print(f"  Not found: {src_base} (skipping)")
        return 0

    total = 0
    for cls_dir in sorted(d for d in src_base.iterdir() if d.is_dir()):
        dst_cls = dst_base / cls_dir.name
        dst_cls.mkdir(parents=True, exist_ok=True)
        images  = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        copies  = compute_aug_copies(len(images)) if augment else 0
        label   = f"  {split_name}/{cls_dir.name}" + (f" +{copies}x" if copies else "")

        for img_path in tqdm(images, desc=label, leave=False):
            try:
                img = open_and_resize(img_path)
                save_image(img, dst_cls / (img_path.stem + ".jpg"))
                total += 1
            except Exception as e:
                print(f"\n  Skip {img_path.name}: {e}")
                continue
            for i in range(copies):
                try:
                    aug = random_augment(open_and_resize(img_path))
                    save_image(aug, dst_cls / f"{img_path.stem}_aug{i+1}.jpg")
                    total += 1
                except Exception as e:
                    print(f"\n  Aug error {img_path.name}: {e}")

    print(f"   {split_name}: {total} images  →  {dst_base}")
    return total


def print_processed_summary():
    base = Path(OUTPUT_DIR) / "processed"
    print("\n" + "=" * 70)
    print("  Processed Dataset Summary (after augmentation)")
    print("=" * 70)
    for s in ("train", "val", "test"):
        d = base / s
        if not d.exists():
            continue
        total  = sum(1 for f in d.rglob("*") if f.suffix == ".jpg")
        n_cls  = sum(1 for x in d.iterdir() if x.is_dir())
        counts = sorted(
            [(x.name, sum(1 for f in x.iterdir() if f.suffix == ".jpg"))
             for x in d.iterdir() if x.is_dir()],
            key=lambda x: x[1],
        )
        if s == "train" and counts:
            mn, mx = counts[0][1], counts[-1][1]
            ratio  = mx / mn if mn > 0 else float("inf")
            print(f"  {s:<6}: {total:>5} images | {n_cls} classes "
                  f"| min {mn} / max {mx} | imbalance {ratio:.1f}x")
        else:
            print(f"  {s:<6}: {total:>5} images | {n_cls} classes")

        # Per-class detail for train
        if s == "train":
            for name, cnt in sorted(counts, key=lambda x: -x[1]):
                print(f"         {name:<40} {cnt:>5}")
    print("=" * 70)
    print(f"\n  Processed path: {base}")


if __name__ == "__main__":
    print("=" * 70)
    print("  split_and_preprocess_v3.py  (70/20/10 | TARGET=350 | anti-overfit)")
    print("=" * 70)
    print(f"\n  TARGET_PER_CLASS={TARGET_PER_CLASS}  MAX_COPIES={MAX_COPIES}")

    print("\n[1/4] Splitting  70 / 20 / 10 ...")
    totals, per_class = split_images()
    print_split_summary(totals, per_class)

    print("\n[2/4] Preprocessing TRAIN (resize + targeted augmentation) ...")
    preprocess_split("train", augment=True)

    print("\n[3/4] Preprocessing VAL  (resize only — no augmentation) ...")
    preprocess_split("val", augment=False)

    print("\n[4/4] Preprocessing TEST (resize only — no augmentation) ...")
    preprocess_split("test", augment=False)

    print_processed_summary()
    print("\n  Done.  Next → python multimodel1_efficientnet_bert.py")