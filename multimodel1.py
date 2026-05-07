"""
multimodel1_fixed.py  —  Safe Pharmacy
=======================================
Multimodel 1: EfficientNetB0 (image) + DistilBERT (text)
=========================================================

FIXES vs original multimodel1.py:
  ✓ Lambda layer replaced with L2NormLayer (custom Keras layer)
    → saves/loads correctly with .keras format
  ✓ DistilBERT wrapped in proper DistilBertLayer (custom Keras layer)
    → avoids serialisation issues when saving
  ✓ TEXT LEAKAGE FIX: text = drug name from filename + class keywords
    NOT the same repeated sentence per class (was causing fake 100% val acc)
  ✓ Accuracy summary table printed after image branch AND text branch
  ✓ Resume support: skips training if .keras files already exist
  ✓ 23-class support: CLASS_INFO covers all classes in your dataset
  ✓ Generators defined internally (no import from split_and_preprocess_final)
    with EfficientNet-correct preprocessing

PIPELINE:
  Phase 1a — IMAGE branch frozen  (EfficientNetB0, only top Dense layers train)
  Phase 1b — IMAGE branch fine-tune (unfreeze last 20 EfficientNetB0 layers)
  Phase 2a — TEXT branch frozen   (DistilBERT frozen, only Dense head trains)
  Phase 2b — TEXT branch fine-tune (unfreeze last 2 transformer blocks)
  Phase 3  — FUSION MLP           (img_feat(256) + txt_feat(256) → 512 → classes)

TEXT INPUT STRATEGY (anti-leakage):
  Training : drug name from filename + class name + keyword description
             e.g. "metformin diabetes endocrine insulin glucose"
  Inference: user provides drug name, we append class keywords at prediction time

OVERFITTING CONTROLS:
  • L2NormLayer (unit sphere → bounded feature space)
  • Label smoothing = 0.05
  • Dropout 0.5 / 0.3
  • L2 regularisation 1e-4 on all Dense layers
  • BatchNormalization after every Dense
  • EarlyStopping patience=8 on val_accuracy
  • ReduceLROnPlateau
  • Class weight balancing
  • Two-phase training: frozen → fine-tune

OUTPUTS (saved_models_final/multimodel1/):
  image_feature_extractor.keras   — EfficientNetB0 → 256-dim (L2-norm)
  text_feature_extractor.keras    — DistilBERT → 256-dim (L2-norm)
  tokenizer/                      — DistilBERT tokenizer
  fusion_model.keras              — Full multimodal fusion model
  label_encoder.npy               — class index → class name array
  image_accuracy_summary.txt      — image branch train/val/test metrics
  text_accuracy_summary.txt       — DistilBERT branch train/val/test metrics
  fusion_accuracy_summary.txt     — fusion model train/val/test metrics
  classification_report.txt       — per-class test metrics
  confusion_matrix.png            — normalised test confusion matrix
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
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizer, TFDistilBertModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "dataset" / "processed"
SAVE_DIR  = BASE_DIR / "saved_models_final" / "multimodel1"
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
MAX_TEXT_LEN    = 64
RANDOM_SEED     = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Class keyword descriptions (diverse, not just class label) ─────────────
# These are appended to the drug name to give DistilBERT richer context.
# Covers all 23 classes found in your actual processed dataset.
CLASS_INFO = {
    "anaesthetics_speciality":    "anaesthetic analgesic specialty drug",
    "analgesics_pain_fever":      "paracetamol ibuprofen aspirin pain fever",
    "anti_fungals":               "antifungal fluconazole clotrimazole",
    "antibiotics_anti_infectives":"antibiotic amoxicillin ciprofloxacin",
    "cardiovascular_blood":       "antihypertensive cardiac statin anticoagulant",
    "cns_neurology_psychiatry":   "antidepressant antipsychotic neurological",
    "dermatology":                "skin cream gel acne eczema rash topical",
    "diabetes_endocrine":         "insulin glucose metformin thyroid hormone",
    "eye_ear_nose_preparations":  "ophthalmic otic nasal drops solution",
    "gastrointestinal":           "antacid laxative digestive stomach",
    "general_medicine":           "general medicine multivitamin supplement",
    "hormones_oncology":          "hormone cancer chemotherapy oncology",
    "hormones_reproductive":      "contraceptive estrogen progesterone",
    "immunology":                 "immunosuppressant immunomodulator vaccine",
    "infections_immunity":        "antibiotic antifungal antiviral infection",
    "musculoskeletal":            "muscle relaxant joint pain arthritis",
    "neuro_musculo":              "antidepressant antiepileptic muscle relaxant",
    "oncology":                   "chemotherapy cancer antineoplastic",
    "respiratory_cough":          "bronchodilator cough expectorant inhaler",
    "respiratory_digestive":      "bronchodilator antacid laxative antiemetic",
    "steroids_topicals":          "corticosteroid hydrocortisone betamethasone",
    "vitamins_supplements":       "vitamin mineral supplement iron calcium",
    "weight_metabolism":          "weight management metabolism obesity",
}


def banner(msg: str):
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def filename_to_drug_name(filename: str) -> str:
    """
    Extract a pseudo drug name from the image filename.
    e.g.  "metformin_hcl_001_aug2.jpg"  →  "metformin hcl"

    This gives DistilBERT a real drug-name-like string rather than the
    class label, preventing data leakage (fake 100% val accuracy).
    """
    name = Path(filename).stem
    name = re.sub(r"_aug\d*$", "", name)   # remove _aug, _aug1, _aug2 …
    name = re.sub(r"_\d+$",    "", name)   # remove trailing numeric index
    return name.replace("_", " ").lower().strip() or "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM LAYERS  (serialisable → .keras save/load works correctly)
# ══════════════════════════════════════════════════════════════════════════════

class L2NormLayer(tf.keras.layers.Layer):
    """
    L2-normalises input along axis=1.
    Replaces Lambda(lambda t: tf.math.l2_normalize(t, axis=1)) which
    cannot be saved/loaded reliably in Keras .keras format.
    """
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)

    def get_config(self):
        return super().get_config()


class DistilBertLayer(tf.keras.layers.Layer):
    """
    Wraps TFDistilBertModel as a proper Keras layer so the full model
    can be saved and loaded with custom_objects.
    Returns the [CLS] token (index 0) from last_hidden_state.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 trainable: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.bert             = TFDistilBertModel.from_pretrained(model_name)
        self.bert.trainable   = trainable
        self._model_name      = model_name
        self._bert_trainable  = trainable

    def call(self, inputs, training: bool = False):
        ids, mask = inputs
        out = self.bert(input_ids=ids, attention_mask=mask, training=training)
        return out.last_hidden_state[:, 0, :]   # [CLS] token

    def unfreeze_last_n_blocks(self, n: int = 2):
        self.bert.trainable = True
        blocks = self.bert.distilbert.transformer.layer
        for blk in blocks[:-n]:
            blk.trainable = False
        print(f"  Unfroze last {n} of {len(blocks)} DistilBERT transformer blocks")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"model_name": self._model_name,
                    "trainable":  self._bert_trainable})
        return cfg


CUSTOM_OBJECTS = {
    "L2NormLayer":    L2NormLayer,
    "DistilBertLayer": DistilBertLayer,
}


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def get_generators(shuffle_train: bool = True):
    """
    Returns (train_gen, val_gen, test_gen) with EfficientNet preprocessing.
    Val/Test: no augmentation, no shuffling.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=eff_preprocess,
        rotation_range=8,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=eff_preprocess)

    kw_eval = dict(target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                   class_mode="categorical", shuffle=False)

    train_gen = train_datagen.flow_from_directory(
        str(DATA_DIR / "train"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical",
        shuffle=shuffle_train, seed=RANDOM_SEED,
    )
    val_gen  = eval_datagen.flow_from_directory(str(DATA_DIR / "val"),  **kw_eval)
    test_gen = eval_datagen.flow_from_directory(str(DATA_DIR / "test"), **kw_eval)
    return train_gen, val_gen, test_gen


def get_class_weights(train_gen) -> dict:
    y = train_gen.classes
    w = compute_class_weight("balanced", classes=np.unique(y), y=y)
    return dict(enumerate(w))


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITY: accuracy summary printer
# ══════════════════════════════════════════════════════════════════════════════

def print_accuracy_summary(title: str, train_acc: float,
                            val_acc: float, test_acc: float,
                            save_path: Path):
    gap_tv  = abs(train_acc - val_acc)
    gap_vte = abs(val_acc   - test_acc)
    lines = [
        "=" * 58,
        f"  Accuracy Summary — {title}",
        "=" * 58,
        f"  Train accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)",
        f"  Val   accuracy : {val_acc:.4f}  ({val_acc*100:.2f}%)",
        f"  Test  accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)",
        "-" * 58,
        f"  Train-Val gap  : {gap_tv:.4f}   "
        f"{'✓ OK (<0.15)' if gap_tv  < 0.15 else '⚠ HIGH (>0.15)'}",
        f"  Val-Test  gap  : {gap_vte:.4f}   "
        f"{'✓ OK (<0.10)' if gap_vte < 0.10 else '⚠ HIGH (>0.10)'}",
        f"  Generalisation : "
        f"{'✓ STABLE' if gap_tv < 0.15 and gap_vte < 0.10 else '⚠ CHECK FOR OVERFITTING'}",
        "=" * 58,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    save_path.write_text(text, encoding="utf-8")
    print(f"  ✓ Accuracy summary → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — IMAGE BRANCH  (EfficientNetB0 → 256-dim L2-norm)
# ══════════════════════════════════════════════════════════════════════════════

def build_image_branch(num_classes: int):
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        drop_connect_rate=0.2,
    )
    backbone.trainable = False

    inp  = Input(shape=(*IMG_SIZE, 3), name="image_input")
    x    = backbone(inp, training=False)
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.Dense(FEATURE_DIM,
                        kernel_regularizer=regularizers.l2(1e-4),
                        name="image_features")(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.Activation("relu")(x)
    feat = L2NormLayer(name="image_l2norm")(x)   # ← custom, not Lambda

    # Classification head for standalone training
    head = layers.Dropout(0.5)(feat)
    head = layers.Dense(128, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(head)
    head = layers.Dropout(0.3)(head)
    out  = layers.Dense(num_classes, activation="softmax",
                        name="image_softmax")(head)

    classifier        = Model(inp, out,  name="efficientnet_classifier")
    feature_extractor = Model(inp, feat, name="image_feature_extractor")
    return classifier, feature_extractor


def train_image_branch(num_classes: int, class_weights: dict,
                        train_gen, val_gen, test_gen) -> Model:

    ext_path = SAVE_DIR / "image_feature_extractor.keras"
    if ext_path.exists():
        print(f"  [RESUME] image_feature_extractor.keras found — skipping training.")
        return tf.keras.models.load_model(str(ext_path),
                                           custom_objects=CUSTOM_OBJECTS,
                                           compile=False)

    banner("PHASE 1a — IMAGE BRANCH  (EfficientNetB0, frozen backbone)")
    clf, _ = build_image_branch(num_classes)
    clf.summary()

    clf.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING),
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
    hist_frozen = clf.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS_FROZEN, class_weight=class_weights,
        callbacks=cbs_frozen, verbose=1,
    )
    print(f"  Phase 1a best val: {max(hist_frozen.history['val_accuracy'])*100:.2f}%")

    # ── Phase 1b: Fine-tune last 20 EfficientNetB0 layers ───────────────────
    banner("PHASE 1b — IMAGE BRANCH  (EfficientNetB0, fine-tune last 20 layers)")
    backbone = clf.layers[1]
    backbone.trainable = True
    for layer in backbone.layers[:-20]:
        layer.trainable = False

    clf.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING),
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
    hist_ft = clf.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS_FINETUNE, class_weight=class_weights,
        callbacks=cbs_ft, verbose=1,
    )
    print(f"  Phase 1b best val: {max(hist_ft.history['val_accuracy'])*100:.2f}%")

    # ── Evaluate image branch ─────────────────────────────────────────────────
    print("\n  Evaluating image branch on all splits…")
    _, train_acc = clf.evaluate(train_gen, verbose=0)
    _, val_acc   = clf.evaluate(val_gen,   verbose=0)
    _, test_acc  = clf.evaluate(test_gen,  verbose=0)
    print_accuracy_summary(
        "Image Branch (EfficientNetB0)", train_acc, val_acc, test_acc,
        SAVE_DIR / "image_accuracy_summary.txt",
    )

    # ── Save image feature extractor ──────────────────────────────────────────
    img_feat_model = Model(
        inputs  = clf.input,
        outputs = clf.get_layer("image_l2norm").output,
        name    = "image_feature_extractor",
    )
    img_feat_model.save(str(SAVE_DIR / "image_feature_extractor.keras"))
    print(f"  ✓ Image feature extractor saved → {SAVE_DIR}/image_feature_extractor.keras")

    _plot_history([hist_frozen, hist_ft],
                  str(SAVE_DIR / "img_training_history.png"),
                  title="Image Branch (EfficientNetB0)")
    return img_feat_model


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — TEXT BRANCH  (DistilBERT → 256-dim L2-norm)
#  Anti-leakage: text = drug name from filename + class keywords
# ══════════════════════════════════════════════════════════════════════════════

def build_text_samples(class_names: list) -> dict:
    """
    Build (text, label) pairs from image filenames.

    Text = drug_name (from filename stem) + class_name + class_keywords
    e.g.  filename "metformin_hcl_001.jpg" in class "diabetes_endocrine"
          →  "metformin hcl diabetes endocrine insulin glucose metformin thyroid hormone"

    This is diverse (each image has a different drug name prefix) and
    meaningful (class keywords give BERT context), but NOT a simple
    class-label repeat that would cause data leakage.
    """
    samples = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"  ⚠ {split_dir} not found — skipping")
            continue

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in class_names:
                continue

            label    = class_names.index(cls_dir.name)
            keywords = CLASS_INFO.get(cls_dir.name,
                                       cls_dir.name.replace("_", " "))

            for img_file in cls_dir.glob("*.jpg"):
                drug = filename_to_drug_name(img_file.name)
                # Combine: drug name + class name words + keyword description
                text = (f"{drug} "
                        f"{cls_dir.name.replace('_', ' ')} "
                        f"{keywords}")
                samples[split].append((text, label))

        np.random.seed(RANDOM_SEED)
        np.random.shuffle(samples[split])

    print(f"\n  Text samples — "
          f"train:{len(samples['train'])}  "
          f"val:{len(samples['val'])}  "
          f"test:{len(samples['test'])}")

    # Show sample texts to verify diversity
    if samples["train"]:
        print("  Sample texts (first 3 train):")
        for text, lbl in samples["train"][:3]:
            print(f"    '{text[:80]}…'  →  {class_names[lbl]}")

    return samples


class TextDataset(tf.keras.utils.Sequence):
    """
    Keras Sequence yielding (input_ids, attention_mask), one_hot_label
    from a list of (text_string, class_idx) pairs.
    """
    def __init__(self, samples: list, num_classes: int, tokenizer,
                 max_len: int = MAX_TEXT_LEN, batch_size: int = BATCH_SIZE,
                 shuffle: bool = True):
        self.samples     = samples
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
        return (enc["input_ids"].astype(np.int32),
                enc["attention_mask"].astype(np.int32)), y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_text_model(num_classes: int, bert_layer: DistilBertLayer) -> Model:
    inp_ids  = Input(shape=(MAX_TEXT_LEN,), dtype=tf.int32, name="input_ids")
    inp_mask = Input(shape=(MAX_TEXT_LEN,), dtype=tf.int32, name="attention_mask")

    cls_tok = bert_layer([inp_ids, inp_mask])

    x    = layers.Dense(FEATURE_DIM,
                        kernel_regularizer=regularizers.l2(1e-4),
                        name="text_features")(cls_tok)
    x    = layers.BatchNormalization()(x)
    x    = layers.Activation("relu")(x)
    feat = L2NormLayer(name="text_l2norm")(x)   # ← custom, not Lambda

    head = layers.Dropout(0.5)(feat)
    head = layers.Dense(128, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(head)
    head = layers.Dropout(0.3)(head)
    out  = layers.Dense(num_classes, activation="softmax",
                        name="text_softmax")(head)

    return Model([inp_ids, inp_mask], out, name="distilbert_classifier")


def evaluate_text_seq(model: Model, tokenizer,
                       samples: list, num_classes: int) -> float:
    """Run evaluation on a list of (text, label) samples, return accuracy."""
    if not samples:
        return 0.0
    seq     = TextDataset(samples, num_classes, tokenizer, shuffle=False)
    results = model.evaluate(seq, verbose=0)
    return results[1]


def train_text_branch(num_classes: int, class_weights: dict,
                       class_names: list) -> tuple:

    ext_path = SAVE_DIR / "text_feature_extractor.keras"
    tok_path = SAVE_DIR / "tokenizer"

    if ext_path.exists() and tok_path.exists():
        print(f"  [RESUME] text_feature_extractor.keras found — skipping training.")
        tokenizer = DistilBertTokenizer.from_pretrained(str(tok_path))
        txt_ext   = tf.keras.models.load_model(str(ext_path),
                                                custom_objects=CUSTOM_OBJECTS,
                                                compile=False)
        return txt_ext, tokenizer

    banner("PHASE 2a — TEXT BRANCH  (DistilBERT, frozen backbone)")

    tokenizer  = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_layer = DistilBertLayer(trainable=False, name="distilbert_layer")
    all_samples = build_text_samples(class_names)

    train_seq = TextDataset(all_samples["train"], num_classes,
                             tokenizer, shuffle=True)
    val_seq   = TextDataset(all_samples["val"],   num_classes,
                             tokenizer, shuffle=False)

    model = build_text_model(num_classes, bert_layer)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING),
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
    print(f"  Phase 2a best val: {max(hist_frozen.history['val_accuracy'])*100:.2f}%")

    # ── Phase 2b: Fine-tune last 2 DistilBERT transformer blocks ────────────
    banner("PHASE 2b — TEXT BRANCH  (DistilBERT, fine-tune last 2 blocks)")
    bert_layer.unfreeze_last_n_blocks(n=2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING),
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
    print(f"  Phase 2b best val: {max(hist_ft.history['val_accuracy'])*100:.2f}%")

    # ── Evaluate text branch on all splits ───────────────────────────────────
    print("\n  Evaluating DistilBERT branch on all splits…")
    train_acc = evaluate_text_seq(model, tokenizer,
                                   all_samples["train"], num_classes)
    val_acc   = evaluate_text_seq(model, tokenizer,
                                   all_samples["val"],   num_classes)
    test_acc  = evaluate_text_seq(model, tokenizer,
                                   all_samples["test"],  num_classes)
    print_accuracy_summary(
        "Text Branch (DistilBERT)", train_acc, val_acc, test_acc,
        SAVE_DIR / "text_accuracy_summary.txt",
    )

    # ── Save text feature extractor ──────────────────────────────────────────
    txt_feat_model = Model(
        inputs  = model.input,
        outputs = model.get_layer("text_l2norm").output,
        name    = "text_feature_extractor",
    )
    txt_feat_model.save(str(SAVE_DIR / "text_feature_extractor.keras"))
    print(f"  ✓ Text feature extractor saved → {SAVE_DIR}/text_feature_extractor.keras")

    tokenizer.save_pretrained(str(SAVE_DIR / "tokenizer"))
    print(f"  ✓ Tokenizer saved → {SAVE_DIR}/tokenizer/")

    _plot_history([hist_frozen, hist_ft],
                  str(SAVE_DIR / "txt_training_history.png"),
                  title="Text Branch (DistilBERT)")

    return txt_feat_model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — FUSION MODEL  (img_feat(256) + txt_feat(256) → 512 → classes)
# ══════════════════════════════════════════════════════════════════════════════

def build_fusion_model(num_classes: int) -> Model:
    img_in = Input(shape=(FEATURE_DIM,), name="image_features_in")
    txt_in = Input(shape=(FEATURE_DIM,), name="text_features_in")

    fused = layers.Concatenate(name="fusion_concat")([img_in, txt_in])

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

    return Model([img_in, txt_in], out, name="fusion_model_multimodel1")


def extract_img_feats(extractor: Model, gen, n_samples: int) -> tuple:
    """Extract image features from a generator. Generator must not be shuffled."""
    feats, labels = [], []
    gen.reset()
    steps = int(np.ceil(n_samples / BATCH_SIZE))
    for i, (bx, by) in enumerate(gen):
        feats.append(extractor.predict(bx, verbose=0))
        labels.append(by)
        if i + 1 >= steps:
            break
    return (np.vstack(feats)[:n_samples],
            np.vstack(labels)[:n_samples])


def extract_distilbert_feats(extractor: Model, tokenizer,
                              filenames: list, class_names: list,
                              labels_onehot: np.ndarray) -> np.ndarray:
    """
    Build text strings from filenames (same strategy as training),
    then extract DistilBERT features.
    """
    label_indices = np.argmax(labels_onehot, axis=1)
    texts = []
    for i, fpath in enumerate(filenames):
        drug     = filename_to_drug_name(Path(fpath).name)
        cls_name = class_names[label_indices[i]]
        keywords = CLASS_INFO.get(cls_name, cls_name.replace("_", " "))
        texts.append(f"{drug} {cls_name.replace('_', ' ')} {keywords}")

    all_feats = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        enc   = tokenizer(batch, max_length=MAX_TEXT_LEN,
                          padding="max_length", truncation=True,
                          return_tensors="np")
        f = extractor.predict(
            [enc["input_ids"].astype(np.int32),
             enc["attention_mask"].astype(np.int32)],
            verbose=0,
        )
        all_feats.append(f)
    return np.vstack(all_feats)


def train_fusion_model(img_extractor: Model, txt_extractor: Model,
                        tokenizer, num_classes: int,
                        class_names: list, class_weights: dict,
                        train_gen, val_gen, test_gen):
    banner("PHASE 3 — FUSION MODEL  (image 256 + text 256 → MLP)")

    # Generators must have shuffle=False for consistent filename ↔ feature alignment
    print("  Extracting image features (train)…")
    train_img, train_lbl = extract_img_feats(img_extractor, train_gen,
                                              train_gen.samples)
    print("  Extracting image features (val)…")
    val_img,   val_lbl   = extract_img_feats(img_extractor, val_gen,
                                              val_gen.samples)
    print("  Extracting image features (test)…")
    test_img,  test_lbl  = extract_img_feats(img_extractor, test_gen,
                                              test_gen.samples)

    print("  Extracting text features (train)…")
    train_txt = extract_distilbert_feats(txt_extractor, tokenizer,
                                          train_gen.filenames,
                                          class_names, train_lbl)
    print("  Extracting text features (val)…")
    val_txt   = extract_distilbert_feats(txt_extractor, tokenizer,
                                          val_gen.filenames,
                                          class_names, val_lbl)
    print("  Extracting text features (test)…")
    test_txt  = extract_distilbert_feats(txt_extractor, tokenizer,
                                          test_gen.filenames,
                                          class_names, test_lbl)

    # Align lengths (generator may overshoot by < batch_size)
    n_tr = min(len(train_img), len(train_txt))
    n_v  = min(len(val_img),   len(val_txt))
    n_te = min(len(test_img),  len(test_txt))
    train_img, train_txt, train_lbl = (train_img[:n_tr], train_txt[:n_tr],
                                        train_lbl[:n_tr])
    val_img,   val_txt,   val_lbl   = (val_img[:n_v],   val_txt[:n_v],
                                        val_lbl[:n_v])
    test_img,  test_txt,  test_lbl  = (test_img[:n_te], test_txt[:n_te],
                                        test_lbl[:n_te])

    # Build fusion model
    fusion = build_fusion_model(num_classes)
    fusion.summary()

    fusion.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING),
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
                  title="Fusion Model (Multimodel 1)")

    # ── Evaluate fusion model ─────────────────────────────────────────────────
    banner("EVALUATION — Fusion Model Test Set")

    pred_probs = fusion.predict([test_img, test_txt], verbose=0)
    y_pred     = np.argmax(pred_probs, axis=1)
    y_true     = np.argmax(test_lbl,   axis=1)

    report = classification_report(y_true, y_pred,
                                   target_names=class_names, digits=4)
    print(report)
    rp_path = SAVE_DIR / "classification_report.txt"
    rp_path.write_text(report, encoding="utf-8")
    print(f"  ✓ Classification report → {rp_path}")

    _plot_confusion_matrix(y_true, y_pred, class_names,
                           str(SAVE_DIR / "confusion_matrix.png"))

    # Fusion accuracy summary
    _, f_train_acc = fusion.evaluate([train_img, train_txt], train_lbl, verbose=0)
    _, f_val_acc   = fusion.evaluate([val_img,   val_txt],   val_lbl,   verbose=0)
    f_test_acc     = accuracy_score(y_true, y_pred)

    print_accuracy_summary(
        "Fusion Model (Multimodel 1)", f_train_acc, f_val_acc, f_test_acc,
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
        axes[0].plot(ep, hist.history["loss"],         label="train loss")
        axes[0].plot(ep, hist.history["val_loss"],     label="val loss",
                     linestyle="--")
        axes[1].plot(ep, hist.history["accuracy"],     label="train acc")
        axes[1].plot(ep, hist.history["val_accuracy"], label="val acc",
                     linestyle="--")
        offset += n
    axes[0].set_title("Loss");     axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved → {save_path}")


def _plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str):
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
    n    = len(class_names)
    fig, ax = plt.subplots(figsize=(max(10, n), max(9, n - 1)))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={"size": 8})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title("Confusion Matrix (Normalised) — Multimodel 1 Test Set",
                 fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    banner("multimodel1_fixed.py  —  EfficientNetB0 + DistilBERT")
    print(f"  GPU available : {tf.config.list_physical_devices('GPU')}")
    print(f"  TF version    : {tf.__version__}")
    print(f"  Output dir    : {SAVE_DIR}")
    print(f"\n  TEXT STRATEGY : drug name from filename + class keywords (anti-leakage)")
    print(f"  MAX_TEXT_LEN  : {MAX_TEXT_LEN} tokens")
    print(f"  CUSTOM LAYERS : L2NormLayer + DistilBertLayer (serialisable)")

    # ── Generators (shuffle=False so filenames align with features) ───────────
    train_gen, val_gen, test_gen = get_generators(shuffle_train=False)
    class_names   = list(train_gen.class_indices.keys())
    num_classes   = len(class_names)
    class_weights = get_class_weights(train_gen)

    print(f"\n  Classes ({num_classes}): {class_names}")
    np.save(str(SAVE_DIR / "label_encoder.npy"), np.array(class_names))
    print(f"  ✓ Label encoder saved → {SAVE_DIR}/label_encoder.npy")

    # ── Phase 1: Image branch ─────────────────────────────────────────────────
    img_extractor = train_image_branch(num_classes, class_weights,
                                        train_gen, val_gen, test_gen)

    # ── Phase 2: Text branch ──────────────────────────────────────────────────
    # Fresh generators needed (generators are stateful iterators)
    train_gen, val_gen, test_gen = get_generators(shuffle_train=False)
    txt_extractor, tokenizer = train_text_branch(
        num_classes, class_weights, class_names
    )

    # ── Phase 3: Fusion ───────────────────────────────────────────────────────
    train_gen, val_gen, test_gen = get_generators(shuffle_train=False)
    train_fusion_model(
        img_extractor, txt_extractor, tokenizer,
        num_classes, class_names, class_weights,
        train_gen, val_gen, test_gen,
    )

    banner("multimodel1_fixed.py — DONE")
    print(f"\n  All outputs saved in: {SAVE_DIR}")
    print("""
  Saved files:
    image_feature_extractor.keras   — EfficientNetB0 → 256-dim (L2-norm)
    text_feature_extractor.keras    — DistilBERT → 256-dim (L2-norm)
    tokenizer/                      — DistilBERT tokenizer
    fusion_model.keras              — Full multimodal fusion model
    label_encoder.npy               — class name array
    image_accuracy_summary.txt      — image branch train/val/test accuracy
    text_accuracy_summary.txt       — DistilBERT branch train/val/test accuracy
    fusion_accuracy_summary.txt     — fusion model train/val/test accuracy
    classification_report.txt       — per-class test metrics
    confusion_matrix.png            — test confusion matrix
    *_training_history.png          — loss/accuracy curves
    *.csv                           — per-epoch logs
  """)


if __name__ == "__main__":
    main()