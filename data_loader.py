# ==============================================================
#  data_loader.py
#  (last updated April 2025 – no changes required for EMA jitter
#   suppression; smoothing is applied inside feature_extractor.py)
# ==============================================================

import os
import re
import json
import numpy as np
from sklearn.utils import resample   # used by oversample_data()

# -----------------------------------------------------------------
# 1.  OpenPose JSON → {frame: [people_keypoints]}
# -----------------------------------------------------------------
def parse_openpose_jsons(openpose_dir):
    """
    Load all *_keypoints.json files from a directory and return:
        { frame_number: [ person0_pose (25×3), person1_pose, … ] }
    If frame numbering in filenames doesn’t start at 0, the dict is
    normalised so the first frame is 0.
    """
    openpose_data = {}
    files = [f for f in os.listdir(openpose_dir) if f.endswith('.json')]

    for fname in sorted(files):
        # ---- extract frame number from filename ----
        frame_num = None
        if '_keypoints' in fname:
            prefix = fname[:fname.rfind('_keypoints')]
            nums = re.findall(r'\d+', prefix)
            if nums:
                frame_num = int(nums[-1])
        else:
            nums = re.findall(r'\d+', fname)
            if nums:
                frame_num = int(nums[-1])
        if frame_num is None:
            continue  # skip unrecognised filenames

        # ---- load JSON ----
        with open(os.path.join(openpose_dir, fname), 'r') as jf:
            data = json.load(jf)

        poses = []
        for person in data.get('people', []):
            keypts = (
                person.get('pose_keypoints_2d')
                or person.get('pose_keypoints')
                or []
            )
            if keypts:
                arr = np.asarray(keypts, dtype=float)
                arr = arr.reshape((-1, 3)) if arr.size % 3 == 0 else np.zeros((0, 3))
            else:
                arr = np.zeros((0, 3))
            poses.append(arr)

        openpose_data[frame_num] = poses

    # normalise so first frame == 0
    if openpose_data:
        first = min(openpose_data.keys())
        if first != 0:
            openpose_data = {fr - first: ppl for fr, ppl in openpose_data.items()}

    return openpose_data

# -----------------------------------------------------------------
# 2.  CVAT COCO‑style JSON → {frame: bbox}
# -----------------------------------------------------------------
def parse_cvat_annotation(cvat_json_path):
    """
    Convert a CVAT COCO‑format annotation file to:
        { frame_number: (x1, y1, x2, y2) }
    """
    with open(cvat_json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data.get('images', [])}
    frame_to_bbox = {}

    for ann in data.get('annotations', []):
        img = images.get(ann['image_id'])
        if not img:
            continue
        frame = int(re.findall(r'\d+', img['file_name'])[-1])
        x, y, w, h = ann['bbox']
        frame_to_bbox[frame] = (int(x), int(y), int(x + w), int(y + h))

    return frame_to_bbox

# -----------------------------------------------------------------
# 3.  Identify which detected skeleton matches the ground‑truth bbox
# -----------------------------------------------------------------
def identify_target_person(people_keypoints, target_bbox,
                           overlap_threshold=0.10, debug=False):
    """
    Returns the index of the skeleton whose bounding box overlaps
    `target_bbox` the most (≥ overlap_threshold).  None if no match.
    """
    x1, y1, x2, y2 = target_bbox
    best_i, best_overlap = None, 0.0

    for i, kps in enumerate(people_keypoints):
        if kps.size == 0:
            continue
        valid = kps[kps[:, 2] > 0.1]
        if valid.size == 0:
            continue
        bx1, by1 = np.min(valid[:, 0]), np.min(valid[:, 1])
        bx2, by2 = np.max(valid[:, 0]), np.max(valid[:, 1])

        inter = max(0, min(x2, bx2) - max(x1, bx1)) \
              * max(0, min(y2, by2) - max(y1, by1))
        area = (bx2 - bx1) * (by2 - by1) + 1e-6
        ov   = inter / area

        if debug:
            print(f"Candidate {i}: overlap={ov:.3f}")
        if ov > best_overlap:
            best_overlap, best_i = ov, i

    return best_i if best_overlap >= overlap_threshold else None

# -----------------------------------------------------------------
# 4.  Turn behaviour time‑intervals into frame‑level labels
# -----------------------------------------------------------------
def label_frames(behavior_intervals, fps=30, total_frames=None):
    """
    Returns {frame: 'class_name'} (default 'none').
    """
    max_t = max(t1 for iv in behavior_intervals.values() for _, t1 in iv)
    total_frames = total_frames or int(max_t * fps) + 1
    labels = {fr: 'none' for fr in range(total_frames)}

    for beh, ivals in behavior_intervals.items():
        for t0, t1 in ivals:
            f0, f1 = int(t0 * fps), int(t1 * fps)
            f1 = min(f1, total_frames - 1)
            for fr in range(f0, f1 + 1):
                labels[fr] = beh
    return labels

# -----------------------------------------------------------------
# 5.  Simple data‑augmentation helpers (unchanged)
# -----------------------------------------------------------------
def augment_data(X, noise_std=0.01, num_augments=2):
    out = []
    for s in X:
        for _ in range(num_augments):
            out.append(s + np.random.normal(0, noise_std, size=s.shape))
    return np.asarray(out)

def oversample_data(X, y, augment=False, noise_std=0.01, num_augments=1):
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    Xo, yo = [], []

    for cls, cnt in zip(unique, counts):
        idx = np.where(y == cls)[0]
        Xc, yc = X[idx], y[idx]

        if cnt < max_count:
            Xr, yr = resample(Xc, yc, replace=True,
                              n_samples=max_count, random_state=42)
            if augment:
                Xa = augment_data(Xc, noise_std, num_augments)
                ya = np.full(Xa.shape[0], cls)
                Xc = np.concatenate([Xr, Xa])[:max_count]
                yc = np.concatenate([yr, ya])[:max_count]
            else:
                Xc, yc = Xr, yr

        Xo.append(Xc); yo.append(yc)

    return np.concatenate(Xo), np.concatenate(yo)
