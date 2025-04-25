# train.py  —  Full pipeline with 5‑Fold CV + final training (April 2025)

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

from data_loader       import (
    parse_openpose_jsons,
    parse_cvat_annotation,
    label_frames,
    oversample_data
)
from feature_extractor import extract_features_for_frames, create_sequences
from model             import build_lstm_model

# ---------------- GPU: allow growth ----------------
try:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# ---------------- custom sparse focal loss ------------
def sparse_focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        nc = tf.shape(y_pred)[1]
        y_true_oh = tf.one_hot(tf.cast(tf.squeeze(y_true, -1), tf.int32), nc)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -y_true_oh * tf.math.log(y_pred)
        fl = alpha * tf.pow(1.0 - y_pred, gamma) * ce
        return tf.reduce_sum(fl, axis=1)
    # ensure the name matches what we pass into custom_objects
    loss_fn.__name__ = "sparse_focal_loss"
    return loss_fn

# ---------------- debug callback ------------------
class PrintCB(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch+1:03d} | loss={logs.get('loss',0):.4f}  "
              f"acc={logs.get('accuracy',0):.4f}  "
              f"val_loss={logs.get('val_loss',0):.4f}  "
              f"val_acc={logs.get('val_accuracy',0):.4f}")

# ---------------- paths & hyper‑params ---------------
OPENPOSE_DIR = "data/openpose_json"
CVAT_JSON    = "data/cvat_labels/annotations.json"
WINDOW       = 30
STRIDE       = 30
BATCH_SIZE   = 16
EPOCHS_CV    = 100
EPOCHS_FULL  = 50
FOLDS        = 5

# --------- behavior intervals (seconds) ------------
behavior_intervals = {
    "Hitting":               [(914, 915), (1245, 1246),
                              (1264, 1266), (1268, 1271),
                              (1278, 1279), (1374, 1376),
                              (2133, 2134), (2616, 2617)],
    "Scratching the Head (SIB)": [(2041, 2046)],
    "Hand Bite (SIB)":        [(1198, 1200.5), (2103, 2106)],
    "Head Hit (SIB)":         [(1200.6, 1203)],
    "Self-Directed Hit (SIB)":[(943, 944), (1416, 1417), (2649, 2650)]
}

def main():
    # === STEP 1: Load data ===
    print("Loading OpenPose JSONs …")
    openpose_data = parse_openpose_jsons(OPENPOSE_DIR)
    print(f"  → {len(openpose_data)} frames found")

    print("Loading CVAT bounding boxes …")
    cvat_bboxes = parse_cvat_annotation(CVAT_JSON)
    print(f"  → {len(cvat_bboxes)} frames with target bbox")

    total_frames = max(openpose_data.keys()) + 1
    frame_labels = label_frames(behavior_intervals, fps=30,
                                total_frames=total_frames)

    # === STEP 2: Feature extraction ===
    print("Extracting engineered features …")
    feat_dict, _ = extract_features_for_frames(
        openpose_data,
        cvat_bboxes,
        include_pose_keypoints=False,
        pca_components=None,
        debug=False
    )
    feat_dim = len(next(iter(feat_dict.values())))
    print(f"Feature dimension per frame: {feat_dim}")

    # === STEP 3: Build sequences & labels ===
    classes = ["none"] + sorted(behavior_intervals.keys())
    cls2idx = {c: i for i, c in enumerate(classes)}
    frame_lbl_idx = {fr: cls2idx[lbl] for fr, lbl in frame_labels.items()}

    seqs, lbls = create_sequences(
        feat_dict,
        labels_dict=frame_lbl_idx,
        window_size=WINDOW,
        stride=STRIDE,
        pad_short_sequences=True
    )
    X = np.asarray(seqs, dtype=float)   # (n_seq, WINDOW, feat_dim)
    y = np.asarray(lbls, dtype=int)     # (n_seq,)

    print(f"Built dataset with {X.shape[0]} sequences of dim {X.shape[1]}×{X.shape[2]}")
    print("Class distribution:", {cls: int((y==i).sum()) for cls,i in cls2idx.items()})

    # === STEP 4: Oversample + augment ===
    X_bal, y_bal = oversample_data(X, y,
                                   augment=True,
                                   noise_std=0.02,
                                   num_augments=2)
    print("After oversampling:", {cls: int((y_bal==i).sum()) for cls,i in cls2idx.items()})

    # === STEP 5: 5‑Fold Cross‑Validation ===
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    agg_cm     = np.zeros((len(classes), len(classes)), dtype=int)
    fold_accs  = []
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_bal, y_bal), 1):
        print(f"\n===== Fold {fold}/{FOLDS} =====")
        X_tr, y_tr = X_bal[train_idx], y_bal[train_idx]
        X_vl, y_vl = X_bal[val_idx], y_bal[val_idx]
        print(f"  Train seqs: {X_tr.shape[0]} | Val seqs: {X_vl.shape[0]}")

        model = build_lstm_model(input_shape=(WINDOW, feat_dim),
                                 num_classes=len(classes))
        model.compile(optimizer="adam",
                      loss=sparse_focal_loss(),
                      metrics=["accuracy"])

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=EPOCHS_CV,
            batch_size=BATCH_SIZE,
            callbacks=[PrintCB(), es],
         #   callbacks=[PrintCB()],
            verbose=2
        )

        y_pred = np.argmax(model.predict(X_vl), axis=1)
        acc = (y_pred == y_vl).mean()
        print(f"Fold {fold} hold‑out accuracy: {acc:.4f}")
        fold_accs.append(acc)

        cm = confusion_matrix(y_vl, y_pred, labels=range(len(classes)))
        agg_cm += cm
        print("Fold confusion matrix:")
        print(cm)

        y_true_all.append(y_vl)
        y_pred_all.append(y_pred)

    # === Aggregated CV results ===
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    print("\n===== Aggregated CV Performance =====")
    print(f"Mean CV accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print("Aggregated confusion matrix:")
    print(agg_cm)
    print("CV classification report:")
    print(classification_report(
        y_true_all, y_pred_all,
        target_names=classes,
        zero_division=0,
        digits=3
    ))

    # === STEP 6: Final training on full dataset ===
    print("\nTraining final model on 100% of data …")
    model_final = build_lstm_model(input_shape=(WINDOW, feat_dim),
                                   num_classes=len(classes))
    model_final.compile(optimizer="adam",
                        loss=sparse_focal_loss(),
                        metrics=["accuracy"])
    model_final.fit(
        X_bal, y_bal,
        epochs=EPOCHS_FULL,
        batch_size=BATCH_SIZE,
        callbacks=[PrintCB()],
        verbose=2
    )

    y_full_pred = np.argmax(model_final.predict(X_bal), axis=1)
    print("\nFinal-model confusion matrix (full dataset):")
    print(confusion_matrix(y_bal, y_full_pred, labels=range(len(classes))))
    print("Final-model classification report (full dataset):")
    print(classification_report(
        y_bal, y_full_pred,
        target_names=classes,
        zero_division=0,
        digits=3
    ))

    # === STEP 7: Save artefacts ===
    model_final.save("behavior_lstm_model.h5")
    with open("classes.json", "w") as cf:
        json.dump(classes, cf)
    print("\n✔  Final model saved to behavior_lstm_model.h5")


if __name__ == "__main__":
    main()
