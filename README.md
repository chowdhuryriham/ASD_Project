# ASD Behavior Detection using Pose Keypoints

This project detects challenging behaviors (e.g., hitting, self-injury) in children with Autism using pose keypoints extracted from videos via OpenPose and processed through an LSTM-based deep learning model.

---

## 🗂 Project Files

- `train.py` – Main training pipeline using 5-fold cross-validation and final training
- `model.py` – Defines the Bidirectional LSTM model with attention
- `feature_extractor.py` – Extracts motion and spatial features from pose keypoints
- `data_loader.py` – Loads OpenPose keypoints and CVAT bounding boxes; assigns behavior labels
- `data_sampling.py` – Performs oversampling and augmentation to handle class imbalance
- `behavior_lstm_model.h5` – Final saved model
- `classes.json` – Class label mappings

---

## ▶️ How to Run

1. Make sure your OpenPose JSON output and CVAT annotations are in the `data/` folder:
   - `data/openpose_json/`
   - `data/cvat_labels/annotations.json`

2. Run the training script:

```bash
python train.py
