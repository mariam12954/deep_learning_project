"""
multimodel2.py  —  Safe Pharmacy  (FIXED)
==========================================
Multimodel 2: MobileNetV2 (image) + BERT (text)
================================================

FIXES FROM v1:
  ✓ TEXT LEAKAGE FIX: Text input is now the DRUG NAME only (from filename),
    NOT the class label. We derive a pseudo drug name from the image filename
    so the text branch must learn visual-language alignment, not just
    memorise class names. At inference, user supplies the actual drug name.
  ✓ 23 classes supported — matches your actual processed dataset.
  ✓ BERT pooler gradient warning suppressed (only [CLS] token used, pooler
    weights excluded from fine-tuning explicitly).
  ✓ Accuracy summary table printed after each branch (image + BERT).
  ✓ Generalization gap check printed for each branch independently.

PIPELINE:
  Phase 1 — IMAGE branch  (MobileNetV2 → 256-dim L2-norm feature vector)
             Frozen → fine-tune last 30 layers
  Phase 2 — TEXT branch   (BERT → 256-dim L2-norm feature vector)
             Frozen → fine-tune last 2 encoder blocks
  Phase 3 — FUSION MLP    (img_feat(256) + txt_feat(256) → 512 → classes)
  Save    → saved_models_final/multimodel2/

TEXT INPUT STRATEGY (anti-leakage):
  Training : extract pseudo drug name from image filename
             e.g.  "metformin_001.jpg"  →  text = "metformin"
             (forces BERT to learn drug name → class mapping, not class→class)
  Inference: user supplies actual drug name typed in, e.g. "Metformin 500mg"

OVERFITTING CONTROLS:
  • Label smoothing = 0.05
  • Class weights (balanced)
  • Early stopping (patience=8)
  • ReduceLROnPlateau
  • L2 regularisation on all Dense layers
  • Dropout 0.5 / 0.3
  • BatchNormalization after every Dense

OUTPUTS (saved_models_final/multimodel2/):
  image_feature_extractor.keras   — MobileNetV2 → 256-dim (L2-norm)
  text_feature_extractor/         — BERT → 256-dim (L2-norm) [SavedModel]
  tokenizer/                      — BERT tokenizer
  fusion_model.keras              — Full multimodal fusion model
  label_encoder.npy               — class index → class name array
  image_accuracy_summary.txt      — image branch train/val/test metrics
  text_accuracy_summary.txt       — BERT branch train/val/test metrics
  classification_report.txt       — fusion test per-class metrics
  confusion_matrix.png            — fusion test confusion matrix
  *_training_history.png          — loss/accuracy curves per branch
  *.csv                           — per-epoch training logs
"""

import os
import re
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, CSVLogger)
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, TFBertModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "dataset" / "processed"
SAVE_DIR  = BASE_DIR / "saved_models_final" / "multimodel2"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16
EPOCHS_FROZEN   = 15
EPOCHS_FINETUNE = 20
LR_FROZEN       = 1e-3
LR_FINETUNE     = 5e-5
LABEL_SMOOTHING = 0.05
FEATURE_DIM     = 256
MAX_TEXT_LEN    = 32          # drug names are short — 32 tokens is enough
RANDOM_SEED     = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── 23-class descriptions (matches your actual processed dataset) ─────────────
CLASS_INFO = {
    "anaesthetics_speciality":    "Anaesthetics and speciality medicines.",
    "analgesics_pain_fever":      "Pain relief and fever reduction.",
    "anti_fungals":               "Antifungal treatments.",
    "antibiotics_anti_infectives":"Antibiotics and anti-infective medicines.",
    "cardiovascular_blood":       "Heart and blood-pressure medicines.",
    "cns_neurology_psychiatry":   "Central nervous system, neurology, psychiatry.",
    "dermatology":                "Skin conditions — acne, eczema, psoriasis.",
    "diabetes_endocrine":         "Blood sugar and endocrine disorders.",
    "eye_ear_nose_preparations":  "Eyes, ears, nose treatments.",
    "gastrointestinal":           "Gastrointestinal and digestive treatments.",
    "general_medicine":           "General medicine and miscellaneous drugs.",
    "hormones_oncology":          "Hormones, oncology, cancer therapy.",
    "hormones_reproductive":      "Reproductive hormones and contraceptives.",
    "immunology":                 "Immunology and immunosuppressants.",
    "infections_immunity":        "Infections and immunity boosters.",
    "musculoskeletal":            "Musculoskeletal and joint medicines.",
    "neuro_musculo":              "Neurology, psychiatry, musculoskeletal.",
    "oncology":                   "Cancer therapy and oncology drugs.",
    "respiratory_cough":          "Respiratory and cough treatments.",
    "respiratory_digestive":      "Respiratory and digestive treatments.",
    "steroids_topicals":          "Topical anti-inflammatory steroids.",
    "vitamins_supplements":       "Vitamins and mineral supplements.",
    "weight_metabolism":          "Weight management and metabolism drugs.",
}


def banner(msg: str):
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def filename_to_drug_name(filename: str) -> str:
    """
    Convert image filename to a pseudo drug name for BERT input.
    e.g.  "metformin_hcl_001_aug2.jpg"  →  "metformin hcl"

    Strategy:
      1. Strip extension and augmentation suffix (_aug\d+)
      2. Strip trailing numeric index (_\d+)
      3. Replace underscores with spaces
      4. Lowercase

    This gives BERT a real drug-name-like string, NOT the class label.
    The model must learn: "metformin hcl" belongs to "diabetes_endocrine".
    """
    name = Path(filename).stem
    name = re.sub(r"_aug\d+$", "", name)
    name = re.sub(r"_\d+$",   "", name)
    name = name.replace("_", " ").lower().strip()
    return name if name else "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — IMAGE BRANCH  (MobileNetV2 → 256-dim)
# ══════════════════════════════════════════════════════════════════════════════

def build_image_branch(num_classes: int):
    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        alpha=1.0,
    )
    backbone.trainable = False

    inp  = Input(shape=(*IMG_SIZE, 3), name="image_input")
    x    = backbone(inp, training=False)
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.Dense(FEATURE_DIM, kernel_regularizer=regularizers.l2(1e-4),
                         name="image_features")(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.Activation("relu")(x)
    feat = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),
                          name="image_l2norm")(x)

    head = layers.Dropout(0.5)(feat)
    head = layers.Dense(128, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(head)
    head = layers.Dropout(0.3)(head)
    out  = layers.Dense(num_classes, activation="softmax",
                        name="image_softmax")(head)

    full_model        = Model(inp, out,  name="mobilenetv2_classifier")
    feature_extractor = Model(inp, feat, name="image_feature_extractor")
    return full_model, feature_extractor


def _print_branch_accuracy_summary(title: str, train_acc: float,
                                    val_acc: float, test_acc: float,
                                    save_path: Path):
    gap_tv  = abs(train_acc - val_acc)
    gap_vte = abs(val_acc   - test_acc)
    lines = [
        "=" * 55,
        f"  Accuracy Summary — {title}",
        "=" * 55,
        f"  Train accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)",
        f"  Val   accuracy : {val_acc:.4f}  ({val_acc*100:.2f}%)",
        f"  Test  accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)",
        "-" * 55,
        f"  Train-Val gap  : {gap_tv:.4f}   {'✓ OK (<0.15)' if gap_tv < 0.15 else '⚠ HIGH (>0.15)'}",
        f"  Val-Test gap   : {gap_vte:.4f}   {'✓ OK (<0.10)' if gap_vte < 0.10 else '⚠ HIGH (>0.10)'}",
        f"  Generalisation : {'✓ STABLE' if gap_tv < 0.15 and gap_vte < 0.10 else '⚠ CHECK FOR OVERFITTING'}",
        "=" * 55,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    save_path.write_text(text, encoding="utf-8")
    print(f"  ✓ Accuracy summary → {save_path}")


def train_image_branch(num_classes: int, class_weights: dict,
                        train_gen, val_gen, test_gen) -> Model:
    banner("PHASE 1a — IMAGE BRANCH  (MobileNetV2, frozen backbone)")

    full_model, _ = build_image_branch(num_classes)
    full_model.summary()

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    cbs_frozen = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(SAVE_DIR / "img_best_frozen.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=0),
        CSVLogger(str(SAVE_DIR / "img_log_frozen.csv")),
    ]

    hist_frozen = full_model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS_FROZEN, class_weight=class_weights,
        callbacks=cbs_frozen, verbose=1,
    )

    # ── Phase 1b: Fine-tune last 30 layers ──────────────────────────────────
    banner("PHASE 1b — IMAGE BRANCH  (MobileNetV2, fine-tune last 30 layers)")
    backbone = full_model.layers[1]
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    cbs_ft = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(SAVE_DIR / "img_best_finetune.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=0),
        CSVLogger(str(SAVE_DIR / "img_log_finetune.csv")),
    ]

    hist_ft = full_model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS_FINETUNE, class_weight=class_weights,
        callbacks=cbs_ft, verbose=1,
    )

    # ── Evaluate image branch on test set ────────────────────────────────────
    print("\n  Evaluating image branch on test set…")
    train_loss, train_acc = full_model.evaluate(train_gen, verbose=0)
    val_loss,   val_acc   = full_model.evaluate(val_gen,   verbose=0)
    test_loss,  test_acc  = full_model.evaluate(test_gen,  verbose=0)

    _print_branch_accuracy_summary(
        "Image Branch (MobileNetV2)", train_acc, val_acc, test_acc,
        SAVE_DIR / "image_accuracy_summary.txt",
    )

    # ── Save feature extractor ────────────────────────────────────────────────
    img_feat_model = tf.keras.Model(
        inputs  = full_model.input,
        outputs = full_model.get_layer("image_l2norm").output,
        name    = "image_feature_extractor",
    )
    img_feat_model.save(str(SAVE_DIR / "image_feature_extractor.keras"))
    print(f"  ✓ Image feature extractor saved → {SAVE_DIR}/image_feature_extractor.keras")

    _plot_history([hist_frozen, hist_ft],
                  str(SAVE_DIR / "img_training_history.png"),
                  title="Image Branch (MobileNetV2)")
    return img_feat_model


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — TEXT BRANCH  (BERT → 256-dim)
#  Anti-leakage: text = drug name from filename, NOT class label
# ══════════════════════════════════════════════════════════════════════════════

class TextDataset(tf.keras.utils.Sequence):
    """
    Yields (input_ids, attention_mask, token_type_ids), one_hot_label.
    Text strings are drug names extracted from image filenames.
    """
    def __init__(self, samples, num_classes, tokenizer,
                 max_len=MAX_TEXT_LEN, batch_size=BATCH_SIZE, shuffle=True):
        self.samples     = samples      # list of (drug_name_str, class_idx)
        self.num_classes = num_classes
        self.tokenizer   = tokenizer
        self.max_len     = max_len
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.indices     = np.arange(len(samples))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:
                                  (idx + 1) * self.batch_size]
        texts  = [self.samples[i][0] for i in batch_idx]
        labels = [self.samples[i][1] for i in batch_idx]

        enc = self.tokenizer(
            texts, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="np",
        )
        y = tf.keras.utils.to_categorical(labels, self.num_classes)
        return (
            enc["input_ids"].astype(np.int32),
            enc["attention_mask"].astype(np.int32),
            enc["token_type_ids"].astype(np.int32),
        ), y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_text_samples_from_files(data_dir: Path,
                                   class_names: list) -> dict:
    """
    Walk processed/{train,val,test}/<class>/*.jpg and build text samples.
    Text = drug name derived from filename (NOT class label).
    Returns dict: {"train": [...], "val": [...], "test": [...]}
    """
    split_samples = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  ⚠ {split_dir} not found — skipping")
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in class_names:
                continue
            idx = class_names.index(cls_dir.name)
            for img_file in cls_dir.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    drug_name = filename_to_drug_name(img_file.name)
                    split_samples[split].append((drug_name, idx))
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(split_samples[split])
    return split_samples


def build_text_branch(num_classes: int) -> Model:
    """BERT [CLS] token → Dense(256) → L2-norm → classifier head"""
    input_ids      = Input(shape=(MAX_TEXT_LEN,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(MAX_TEXT_LEN,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(MAX_TEXT_LEN,), dtype=tf.int32, name="token_type_ids")

    bert = TFBertModel.from_pretrained("bert-base-uncased")
    bert.trainable = False

    bert_out  = bert(input_ids, attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     output_hidden_states=False)
    # Use [CLS] token from last_hidden_state — avoids pooler gradient warning
    cls_token = bert_out.last_hidden_state[:, 0, :]

    x    = layers.Dense(FEATURE_DIM, kernel_regularizer=regularizers.l2(1e-4),
                         name="text_features")(cls_token)
    x    = layers.BatchNormalization()(x)
    x    = layers.Activation("relu")(x)
    feat = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),
                          name="text_l2norm")(x)

    head = layers.Dropout(0.5)(feat)
    head = layers.Dense(128, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(head)
    head = layers.Dropout(0.3)(head)
    out  = layers.Dense(num_classes, activation="softmax",
                        name="text_softmax")(head)

    model = Model([input_ids, attention_mask, token_type_ids], out,
                  name="bert_classifier")
    return model


def evaluate_text_branch(model: Model, tokenizer,
                          split_samples: dict, num_classes: int,
                          split_name: str) -> float:
    """Run inference on a split and return accuracy."""
    samples = split_samples[split_name]
    if not samples:
        return 0.0
    seq = TextDataset(samples, num_classes, tokenizer,
                      shuffle=False, batch_size=BATCH_SIZE)
    results = model.evaluate(seq, verbose=0)
    return results[1]   # accuracy


def train_text_branch(num_classes: int, class_weights: dict,
                       class_names: list) -> tuple:
    banner("PHASE 2a — TEXT BRANCH  (BERT, frozen backbone)")

    tokenizer    = BertTokenizer.from_pretrained("bert-base-uncased")
    split_samples = build_text_samples_from_files(DATA_DIR, class_names)

    n_train = len(split_samples["train"])
    n_val   = len(split_samples["val"])
    n_test  = len(split_samples["test"])
    print(f"\n  Text samples — train: {n_train}  val: {n_val}  test: {n_test}")
    print(f"  Sample texts (first 3 train):")
    for text, lbl in split_samples["train"][:3]:
        print(f"    '{text}'  →  class={class_names[lbl]}")

    train_seq = TextDataset(split_samples["train"], num_classes,
                             tokenizer, shuffle=True)
    val_seq   = TextDataset(split_samples["val"],   num_classes,
                             tokenizer, shuffle=False)

    model = build_text_branch(num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    cbs_frozen = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(SAVE_DIR / "txt_best_frozen.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=0),
        CSVLogger(str(SAVE_DIR / "txt_log_frozen.csv")),
    ]

    hist_frozen = model.fit(
        train_seq, validation_data=val_seq,
        epochs=EPOCHS_FROZEN, class_weight=class_weights,
        callbacks=cbs_frozen, verbose=1,
    )

    # ── Phase 2b: Fine-tune last 2 BERT encoder blocks ──────────────────────
    banner("PHASE 2b — TEXT BRANCH  (BERT, fine-tune last 2 encoder blocks)")

    bert_layer = None
    for layer in model.layers:
        if isinstance(layer, TFBertModel):
            bert_layer = layer
            break

    if bert_layer is not None:
        bert_layer.trainable = True
        # Freeze everything except last 2 transformer encoder blocks
        # and explicitly freeze pooler to suppress gradient warning
        for enc_block in bert_layer.bert.encoder.layer[:-2]:
            enc_block.trainable = False
        bert_layer.bert.pooler.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    cbs_ft = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(SAVE_DIR / "txt_best_finetune.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=0),
        CSVLogger(str(SAVE_DIR / "txt_log_finetune.csv")),
    ]

    hist_ft = model.fit(
        train_seq, validation_data=val_seq,
        epochs=EPOCHS_FINETUNE, class_weight=class_weights,
        callbacks=cbs_ft, verbose=1,
    )

    # ── Evaluate text branch on all splits ───────────────────────────────────
    print("\n  Evaluating BERT branch on all splits…")
    train_acc = evaluate_text_branch(model, tokenizer, split_samples,
                                      num_classes, "train")
    val_acc   = evaluate_text_branch(model, tokenizer, split_samples,
                                      num_classes, "val")
    test_acc  = evaluate_text_branch(model, tokenizer, split_samples,
                                      num_classes, "test")

    _print_branch_accuracy_summary(
        "Text Branch (BERT)", train_acc, val_acc, test_acc,
        SAVE_DIR / "text_accuracy_summary.txt",
    )

    # ── Save text feature extractor ──────────────────────────────────────────
    txt_feat_model = tf.keras.Model(
        inputs  = model.input,
        outputs = model.get_layer("text_l2norm").output,
        name    = "text_feature_extractor",
    )
    txt_feat_model.save(str(SAVE_DIR / "text_feature_extractor"))
    print(f"  ✓ Text feature extractor saved → {SAVE_DIR}/text_feature_extractor/")

    tokenizer.save_pretrained(str(SAVE_DIR / "tokenizer"))
    print(f"  ✓ Tokenizer saved → {SAVE_DIR}/tokenizer/")

    _plot_history([hist_frozen, hist_ft],
                  str(SAVE_DIR / "txt_training_history.png"),
                  title="Text Branch (BERT)")

    return txt_feat_model, tokenizer, split_samples


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — FUSION MODEL  (img_feat(256) + txt_feat(256) → 512 → classes)
# ══════════════════════════════════════════════════════════════════════════════

def build_fusion_model(num_classes: int) -> Model:
    img_input = Input(shape=(FEATURE_DIM,), name="image_features_in")
    txt_input = Input(shape=(FEATURE_DIM,), name="text_features_in")

    fused = layers.Concatenate(name="fusion_concat")([img_input, txt_input])

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4),
                     name="mlp_256")(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4),
                     name="mlp_128")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(num_classes, activation="softmax",
                       name="fusion_output")(x)

    model = Model([img_input, txt_input], out, name="fusion_model_multimodel2")
    return model


def extract_image_feats(img_extractor: Model, gen, n_samples: int) -> tuple:
    """Extract image feature vectors from a Keras generator."""
    feats, labels = [], []
    steps = int(np.ceil(n_samples / BATCH_SIZE))
    for i, (batch_x, batch_y) in enumerate(gen):
        feats.append(img_extractor.predict(batch_x, verbose=0))
        labels.append(batch_y)
        if i + 1 >= steps:
            break
    return np.vstack(feats)[:n_samples], np.vstack(labels)[:n_samples]


def extract_text_feats(txt_extractor: Model, tokenizer,
                        samples: list) -> np.ndarray:
    """Extract BERT feature vectors from (drug_name, label) sample list."""
    all_feats = []
    for i in range(0, len(samples), BATCH_SIZE):
        batch_texts = [s[0] for s in samples[i: i + BATCH_SIZE]]
        enc = tokenizer(
            batch_texts, max_length=MAX_TEXT_LEN, padding="max_length",
            truncation=True, return_tensors="np",
        )
        f = txt_extractor.predict(
            [enc["input_ids"].astype(np.int32),
             enc["attention_mask"].astype(np.int32),
             enc["token_type_ids"].astype(np.int32)],
            verbose=0,
        )
        all_feats.append(f)
    return np.vstack(all_feats)


def train_fusion_model(img_extractor: Model, txt_extractor: Model,
                        tokenizer, num_classes: int,
                        class_names: list, class_weights: dict,
                        train_gen, val_gen, test_gen,
                        split_samples: dict):
    banner("PHASE 3 — FUSION MODEL  (image 256 + text 256 → MLP → classes)")

    # Image features from generator
    print("  Extracting image features (train)…")
    train_img, train_lbl = extract_image_feats(img_extractor, train_gen,
                                                train_gen.samples)
    print("  Extracting image features (val)…")
    val_img,   val_lbl   = extract_image_feats(img_extractor, val_gen,
                                                val_gen.samples)
    print("  Extracting image features (test)…")
    test_img,  test_lbl  = extract_image_feats(img_extractor, test_gen,
                                                test_gen.samples)

    # Text features from filename-derived drug names
    # Align with generator order by rebuilding sample lists in generator order
    def gen_order_samples(gen, split_name):
        """
        Rebuild (drug_name, class_idx) list in the same order as the generator.
        gen.filenames gives paths in generator order.
        """
        out = []
        for fpath in gen.filenames:
            fname    = Path(fpath).name
            cls_name = Path(fpath).parent.name
            label    = class_names.index(cls_name) if cls_name in class_names else 0
            drug     = filename_to_drug_name(fname)
            out.append((drug, label))
        return out

    print("  Building text samples aligned with generator order…")
    train_txt_samples = gen_order_samples(train_gen, "train")
    val_txt_samples   = gen_order_samples(val_gen,   "val")
    test_txt_samples  = gen_order_samples(test_gen,  "test")

    print("  Extracting text features (train)…")
    train_txt = extract_text_feats(txt_extractor, tokenizer, train_txt_samples)
    print("  Extracting text features (val)…")
    val_txt   = extract_text_feats(txt_extractor, tokenizer, val_txt_samples)
    print("  Extracting text features (test)…")
    test_txt  = extract_text_feats(txt_extractor, tokenizer, test_txt_samples)

    # Align lengths
    n_tr = min(len(train_img), len(train_txt))
    n_v  = min(len(val_img),   len(val_txt))
    n_te = min(len(test_img),  len(test_txt))
    train_img, train_txt, train_lbl = (train_img[:n_tr], train_txt[:n_tr],
                                        train_lbl[:n_tr])
    val_img,   val_txt,   val_lbl   = (val_img[:n_v],   val_txt[:n_v],
                                        val_lbl[:n_v])
    test_img,  test_txt,  test_lbl  = (test_img[:n_te], test_txt[:n_te],
                                        test_lbl[:n_te])

    # Build and train fusion model
    fusion = build_fusion_model(num_classes)
    fusion.summary()

    fusion.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    cbs = [
        EarlyStopping(monitor="val_accuracy", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(SAVE_DIR / "fusion_best.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=0),
        CSVLogger(str(SAVE_DIR / "fusion_log.csv")),
    ]

    hist = fusion.fit(
        [train_img, train_txt], train_lbl,
        validation_data=([val_img, val_txt], val_lbl),
        epochs=40, batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cbs, verbose=1,
    )

    fusion.save(str(SAVE_DIR / "fusion_model.keras"))
    print(f"  ✓ Fusion model saved → {SAVE_DIR}/fusion_model.keras")
    _plot_history([hist], str(SAVE_DIR / "fusion_training_history.png"),
                  title="Fusion Model (Multimodel 2)")

    # ── Evaluate fusion on test set ──────────────────────────────────────────
    banner("EVALUATION — Fusion Model Test Set")

    pred_probs = fusion.predict([test_img, test_txt], verbose=0)
    y_pred     = np.argmax(pred_probs,  axis=1)
    y_true     = np.argmax(test_lbl,    axis=1)

    report = classification_report(y_true, y_pred,
                                   target_names=class_names, digits=4)
    print(report)
    rp_path = SAVE_DIR / "classification_report.txt"
    rp_path.write_text(report, encoding="utf-8")
    print(f"  ✓ Classification report → {rp_path}")

    _plot_confusion_matrix(y_true, y_pred, class_names,
                           str(SAVE_DIR / "confusion_matrix.png"))

    # Fusion accuracy summary
    f_train_loss, f_train_acc = fusion.evaluate([train_img, train_txt],
                                                 train_lbl, verbose=0)
    f_val_loss,   f_val_acc   = fusion.evaluate([val_img,   val_txt],
                                                 val_lbl,   verbose=0)
    f_test_acc = accuracy_score(y_true, y_pred)

    _print_branch_accuracy_summary(
        "Fusion Model (Multimodel 2)", f_train_acc, f_val_acc, f_test_acc,
        SAVE_DIR / "fusion_accuracy_summary.txt",
    )

    return fusion


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _plot_history(histories: list, save_path: str, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    offset = 0
    for hist in histories:
        n  = len(hist.history["loss"])
        ep = range(offset, offset + n)
        axes[0].plot(ep, hist.history["loss"],         label="train_loss")
        axes[0].plot(ep, hist.history["val_loss"],     label="val_loss",
                     linestyle="--")
        axes[1].plot(ep, hist.history["accuracy"],     label="train_acc")
        axes[1].plot(ep, hist.history["val_accuracy"], label="val_acc",
                     linestyle="--")
        offset += n
    axes[0].set_title("Loss");     axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved → {save_path}")


def _plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str):
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
    n    = len(class_names)
    fig_sz = max(10, n)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz - 1))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={"size": 8})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title("Confusion Matrix (Normalised) — Multimodel 2 Test Set",
                 fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix → {save_path}")


def get_class_weights(train_gen) -> dict:
    labels  = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(enumerate(weights))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    banner("multimodel2.py  (FIXED)  —  MobileNetV2 + BERT")
    print(f"  GPU available : {tf.config.list_physical_devices('GPU')}")
    print(f"  TF version    : {tf.__version__}")
    print(f"  Output dir    : {SAVE_DIR}")
    print(f"\n  TEXT STRATEGY : drug name from filename (anti-leakage)")
    print(f"  MAX_TEXT_LEN  : {MAX_TEXT_LEN} tokens")

    import sys
    sys.path.insert(0, str(BASE_DIR))
    from split_and_preprocess_final import get_generators

    # ── Generators ────────────────────────────────────────────────────────────
    train_gen, val_gen, test_gen = get_generators(backbone="mobilenet")
    class_names   = list(train_gen.class_indices.keys())
    num_classes   = len(class_names)
    class_weights = get_class_weights(train_gen)

    print(f"\n  Classes ({num_classes}): {class_names}")
    np.save(str(SAVE_DIR / "label_encoder.npy"), np.array(class_names))
    print(f"  ✓ Label encoder saved → {SAVE_DIR}/label_encoder.npy")

    # ── Phase 1: Image branch ─────────────────────────────────────────────────
    img_extractor = train_image_branch(num_classes, class_weights,
                                        train_gen, val_gen, test_gen)

    # Reset generators
    train_gen, val_gen, test_gen = get_generators(backbone="mobilenet")

    # ── Phase 2: Text branch ──────────────────────────────────────────────────
    txt_extractor, tokenizer, split_samples = train_text_branch(
        num_classes, class_weights, class_names
    )

    # Reset generators for fusion
    train_gen, val_gen, test_gen = get_generators(backbone="mobilenet")

    # ── Phase 3: Fusion ───────────────────────────────────────────────────────
    train_fusion_model(
        img_extractor, txt_extractor, tokenizer,
        num_classes, class_names, class_weights,
        train_gen, val_gen, test_gen,
        split_samples,
    )

    banner("multimodel2.py — DONE")
    print(f"\n  All outputs saved in: {SAVE_DIR}")
    print("""
  Saved files:
    image_feature_extractor.keras   — MobileNetV2 → 256-dim (L2-norm)
    text_feature_extractor/         — BERT → 256-dim (L2-norm) [SavedModel]
    tokenizer/                      — BERT tokenizer
    fusion_model.keras              — Full multimodal fusion model
    label_encoder.npy               — class name array
    image_accuracy_summary.txt      — image branch train/val/test accuracy
    text_accuracy_summary.txt       — BERT branch train/val/test accuracy
    fusion_accuracy_summary.txt     — fusion model train/val/test accuracy
    classification_report.txt       — fusion per-class test metrics
    confusion_matrix.png            — fusion test confusion matrix
    *_training_history.png          — loss/accuracy curves
    *.csv                           — per-epoch logs
  """)


if __name__ == "__main__":
    main()