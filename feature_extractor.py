import math
import numpy as np
import pickle
from sklearn.decomposition import PCA
from smoothing import KeypointEMA            # --- NEW ---

# ---------------- BODY_25 indices ----------------
NOSE, NECK = 0, 1
RS, RE, RW = 2, 3, 4
LS, LE, LW = 5, 6, 7
RHIP, LHIP   = 9, 12
CONF_THR     = 0.10

# ---------------- global EMA for offline pass ----------------
_ema_global = KeypointEMA(alpha=0.30, conf_thr=0.15)   # --- EMA ---

# ==============================================================
#  Helper utilities
# ==============================================================

def _coord(kps, idx):
    """Return (x,y) or None if missing/low‑conf."""
    if kps is None or idx >= kps.shape[0] or kps[idx, 2] < CONF_THR:
        return None
    return (kps[idx, 0], kps[idx, 1])

def _dist(A, B):
    return 0.0 if A is None or B is None else math.hypot(A[0]-B[0], A[1]-B[1])

def _angle(A, B, C):
    """Angle at B (degrees)."""
    if A is None or B is None or C is None:
        return 0.0
    v1 = (A[0]-B[0], A[1]-B[1])
    v2 = (C[0]-B[0], C[1]-B[1])
    d1 = math.hypot(*v1)
    d2 = math.hypot(*v2)
    if d1 < 1e-6 or d2 < 1e-6:
        return 0.0
    cosv = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1])/(d1*d2)))
    return math.degrees(math.acos(cosv))

# ==============================================================
#  Core per‑frame feature routine
# ==============================================================

def compute_frame_features(frame_num,
                           tgt_kps,
                           other_kps_list,
                           prev_tgt_kps=None,
                           prev_frame_num=None,
                           prev_dists=None,
                           debug=False):
    """
    Returns (feature_vector, new_prev_dists)

    `prev_dists` carries previous distances for closing‑speed calculations.
    """

    # 1. Smooth key‑points to suppress jitter  (EMA)
    tgt_kps = _ema_global(tgt_kps)
    if other_kps_list:
        other_kps_list = [_ema_global(p) for p in other_kps_list]

    # 2. Normalise coordinates (hip‑midpoint = origin)
    rh = _coord(tgt_kps, RHIP); lh = _coord(tgt_kps, LHIP)
    if rh and lh:
        hip_mid = ((rh[0]+lh[0])*0.5, (rh[1]+lh[1])*0.5)
    else:
        hip_mid = rh or lh or (0.0, 0.0)

    norm = [(0.0, 0.0)] * tgt_kps.shape[0]
    for i in range(tgt_kps.shape[0]):
        if tgt_kps[i, 2] >= CONF_THR:
            norm[i] = (tgt_kps[i, 0] - hip_mid[0],
                       tgt_kps[i, 1] - hip_mid[1])

    # 3. Velocities (hand L/R, head)
    df = 1 if prev_frame_num is None else max(1, frame_num - prev_frame_num)
    def vel(idx):
        if prev_tgt_kps is None:
            return 0.0
        return _dist(_coord(prev_tgt_kps, idx), _coord(tgt_kps, idx)) / df

    lhand_vel = vel(LW)
    rhand_vel = vel(RW)
    head_vel = vel(NOSE) if _coord(tgt_kps, NOSE) else vel(NECK)

    # 4. Angles (normalized to [0,1])
    l_elbow = _angle(norm[LW], norm[LE], norm[LS]) / 180.0
    r_elbow = _angle(norm[RW], norm[RE], norm[RS]) / 180.0
    l_swing = _angle(norm[LS], (0.0,0.0), norm[LE]) / 180.0
    r_swing = _angle(norm[RS], (0.0,0.0), norm[RE]) / 180.0

    # 5. Self‑directed distances & closing speeds
    head_pt = norm[NOSE] if _coord(tgt_kps, NOSE) else norm[NECK]
    mouth_pt = head_pt
    d_lh_head  = _dist(norm[LW], head_pt)
    d_rh_head  = _dist(norm[RW], head_pt)
    d_lh_mouth = _dist(norm[LW], mouth_pt)
    d_rh_mouth = _dist(norm[RW], mouth_pt)

    if prev_dists:
        vlh_head  = prev_dists['d_lh_head']  - d_lh_head
        vrh_head  = prev_dists['d_rh_head']  - d_rh_head
        vlh_mouth = prev_dists['d_lh_mouth'] - d_lh_mouth
        vrh_mouth = prev_dists['d_rh_mouth'] - d_rh_mouth
    else:
        vlh_head = vrh_head = vlh_mouth = vrh_mouth = 0.0

    # 6. Interpersonal features (victim selection by hand direction)
    dist_lh_oth_h = dist_rh_oth_h = 0.0
    dist_lh_oth_t = dist_rh_oth_t = 0.0
    vlh_oth_h = vrh_oth_h = vlh_oth_t = vrh_oth_t = 0.0
    ang_lh_app = ang_rh_app = 0.5
    torso_diff = 0.0

    if other_kps_list:
        # 6a. Determine which hand is "active" (faster) and get its motion vector
        use_v = None; hand_pos = None
        if prev_tgt_kps is not None:
            hl_now  = _coord(tgt_kps, LW);  hl_prev = _coord(prev_tgt_kps, LW)
            hr_now  = _coord(tgt_kps, RW);  hr_prev = _coord(prev_tgt_kps, RW)
            v_l = None if hl_now  is None or hl_prev  is None else (hl_now[0]-hl_prev[0], hl_now[1]-hl_prev[1])
            v_r = None if hr_now  is None or hr_prev  is None else (hr_now[0]-hr_prev[0], hr_now[1]-hr_prev[1])
            if v_l and v_r:
                if lhand_vel >= rhand_vel:
                    use_v, hand_pos = v_l, hl_now
                else:
                    use_v, hand_pos = v_r, hr_now
            elif v_l:
                use_v, hand_pos = v_l, hl_now
            elif v_r:
                use_v, hand_pos = v_r, hr_now

        # 6b. Default victim = first skeleton
        oth = other_kps_list[0]

        # 6c. If we have a valid motion vector, pick the other whose head best aligns
        if use_v and hand_pos:
            best_cos = -1.0
            mag_v = math.hypot(use_v[0], use_v[1])
            if mag_v > 1e-6:
                for person in other_kps_list:
                    head_o = _coord(person, NOSE) or _coord(person, NECK)
                    if head_o is None:
                        continue
                    vt = (head_o[0]-hand_pos[0], head_o[1]-hand_pos[1])
                    mag_t = math.hypot(vt[0], vt[1])
                    if mag_t < 1e-6:
                        continue
                    cosv = (use_v[0]*vt[0] + use_v[1]*vt[1])/(mag_v*mag_t)
                    if cosv > best_cos:
                        best_cos, oth = cosv, person

        # 6d. Compute distances & speeds to chosen victim
        oth_head = _coord(oth, NOSE) or _coord(oth, NECK)
        rhip_o, lhip_o = _coord(oth, RHIP), _coord(oth, LHIP)
        if rhip_o and lhip_o:
            oth_torso = ((rhip_o[0]+lhip_o[0])*0.5, (rhip_o[1]+lhip_o[1])*0.5)
        else:
            oth_torso = _coord(oth, NECK)

        dist_lh_oth_h = _dist(_coord(tgt_kps, LW), oth_head)
        dist_rh_oth_h = _dist(_coord(tgt_kps, RW), oth_head)
        dist_lh_oth_t = _dist(_coord(tgt_kps, LW), oth_torso)
        dist_rh_oth_t = _dist(_coord(tgt_kps, RW), oth_torso)

        if prev_dists:
            vlh_oth_h = prev_dists['dist_lh_oth_h'] - dist_lh_oth_h
            vrh_oth_h = prev_dists['dist_rh_oth_h'] - dist_rh_oth_h
            vlh_oth_t = prev_dists['dist_lh_oth_t'] - dist_lh_oth_t
            vrh_oth_t = prev_dists['dist_rh_oth_t'] - dist_rh_oth_t

        # 6e. Approach angles
        def app_ang(hand_now, hand_prev, tgt_pt):
            if hand_now is None or hand_prev is None or tgt_pt is None:
                return 0.5
            vm = (hand_now[0]-hand_prev[0], hand_now[1]-hand_prev[1])
            vt = (tgt_pt[0]-hand_now[0],    tgt_pt[1]-hand_now[1])
            dv, dt = math.hypot(*vm), math.hypot(*vt)
            if dv < 1e-6 or dt < 1e-6:
                return 0.5
            cosv = max(-1.0, min(1.0, (vm[0]*vt[0] + vm[1]*vt[1])/(dv*dt)))
            return math.degrees(math.acos(cosv)) / 180.0

        ang_lh_app = app_ang(_coord(tgt_kps, LW),
                             _coord(prev_tgt_kps, LW) if prev_tgt_kps is not None else None,
                             oth_head)
        ang_rh_app = app_ang(_coord(tgt_kps, RW),
                             _coord(prev_tgt_kps, RW) if prev_tgt_kps is not None else None,
                             oth_head)

        # 6f. Torso‑orientation difference
        def torso_angle(kps):
            rs = _coord(kps, RS); ls = _coord(kps, LS)
            if rs is None or ls is None:
                return None
            return math.degrees(math.atan2(ls[1]-rs[1], ls[0]-rs[0]))

        ang_t = torso_angle(tgt_kps)
        ang_o = torso_angle(oth)
        if ang_t is not None and ang_o is not None:
            diff = abs(ang_t - ang_o)
            torso_diff = (360 - diff if diff > 180 else diff) / 180.0

    # 7. Assemble feature vector
    features = [
        lhand_vel, rhand_vel, head_vel,
        l_elbow,  r_elbow,  l_swing,  r_swing,
        d_lh_head, d_rh_head, d_lh_mouth, d_rh_mouth,
        vlh_head, vrh_head, vlh_mouth, vrh_mouth,
        dist_lh_oth_h, dist_rh_oth_h, dist_lh_oth_t, dist_rh_oth_t,
        vlh_oth_h, vrh_oth_h, vlh_oth_t, vrh_oth_t,
        ang_lh_app, ang_rh_app, torso_diff
    ]

    new_prev = {
        'd_lh_head': d_lh_head,   'd_rh_head': d_rh_head,
        'd_lh_mouth': d_lh_mouth, 'd_rh_mouth': d_rh_mouth,
        'dist_lh_oth_h': dist_lh_oth_h, 'dist_rh_oth_h': dist_rh_oth_h,
        'dist_lh_oth_t': dist_lh_oth_t, 'dist_rh_oth_t': dist_rh_oth_t
    }

    if debug:
        print(f"[Frame {frame_num}] LVel={lhand_vel:.2f} RVel={rhand_vel:.2f} "
              f"L→Head={d_lh_head:.2f} R→Head={d_rh_head:.2f}")

    return features, new_prev

# ==============================================================
#  Bulk extraction & PCA
# ==============================================================
def extract_features_for_frames(openpose_data,
                                target_bboxes,
                                include_pose_keypoints=False,
                                pca_components=None,
                                debug=False):
    """
    Iterate over frames, identify the target person, call
    compute_frame_features, and return:
        features_dict – {frame: feature_vector}
        pca_model     – fitted PCA (or None)
    """
    from data_loader import identify_target_person

    feats, raw_pose, prev_kps, prev_fr, prev_d = {}, {}, None, None, None

    # Optional debug info
    if debug:
        print("\n--- Debug: Frame Numbers ---")
        print(f"OpenPose frames (first 20): {sorted(openpose_data.keys())[:20]}")
        print(f"CVAT frames     (first 20): {sorted(target_bboxes.keys())[:20]}")
        print(f"Total OpenPose frames: {len(openpose_data)}")
        print(f"Total CVAT frames:     {len(target_bboxes)}")
        print("--- End Debug Info ---\n")

    for fr in sorted(openpose_data.keys()):
        if fr not in target_bboxes:
            continue
        people = openpose_data[fr]
        if not people:
            continue

        idx = identify_target_person(people,
                                     target_bboxes[fr],
                                     overlap_threshold=0.10,
                                     debug=debug)
        if idx is None:
            prev_kps = prev_d = None
            continue

        tgt    = people[idx]
        others = [p for j,p in enumerate(people) if j != idx]

        vec, prev_d = compute_frame_features(
            fr, tgt, others,
            prev_tgt_kps=prev_kps,
            prev_frame_num=prev_fr,
            prev_dists=prev_d,
            debug=debug
        )
        feats[fr] = vec
        prev_kps, prev_fr = tgt, fr

        # raw pose for optional PCA
        if include_pose_keypoints:
            coords = []
            for (x,y,c) in tgt:
                coords.extend([x,y] if c > CONF_THR else [0.0,0.0])
            raw_pose[fr] = coords

    # Optional PCA on raw pose coords
    pca_model = None
    if include_pose_keypoints and pca_components:
        pca_model = PCA(n_components=pca_components)
        mat = np.array([raw_pose[fr] for fr in sorted(raw_pose)], float)
        pca_model.fit(mat)
        trans = pca_model.transform(mat)
        for i, fr in enumerate(sorted(raw_pose)):
            feats[fr].extend(list(trans[i]))
    elif include_pose_keypoints:
        for fr, coords in raw_pose.items():
            feats[fr].extend(coords)

    if pca_model:
        with open("pca_model.pkl", "wb") as pf:
            pickle.dump(pca_model, pf)

    return feats, pca_model

# ==============================================================
#  Sliding‑window sequence builder (zero‑padding support)
# ==============================================================
def create_sequences(features_dict,
                     labels_dict=None,
                     window_size=30,
                     stride=30,
                     pad_short_sequences=True):
    """
    Returns (sequences, seq_labels).  Each sequence is a list of
    feature vectors of length == window_size (zero‑padded if needed).
    """
    frames_sorted = sorted(features_dict.keys())
    if not frames_sorted:
        return [], []

    # split into contiguous segments
    segs, seg = [], [frames_sorted[0]]
    for i in range(1, len(frames_sorted)):
        if frames_sorted[i] == frames_sorted[i-1] + 1:
            seg.append(frames_sorted[i])
        else:
            segs.append(seg); seg = [frames_sorted[i]]
    segs.append(seg)

    seqs, lbls = [], []
    feat_dim = len(next(iter(features_dict.values())))

    for seg in segs:
        # full‑length windows
        if len(seg) >= window_size:
            for start in range(0, len(seg)-window_size+1, stride):
                window = seg[start:start+window_size]
                seqs.append([features_dict[f] for f in window])
                if labels_dict is not None:
                    lbls.append(labels_dict.get(window[-1], 0))
        # tail‑padding
        elif pad_short_sequences:
            pad_len = window_size - len(seg)
            seq = [features_dict[f] for f in seg] + [[0.0]*feat_dim]*pad_len
            seqs.append(seq)
            if labels_dict is not None:
                lbls.append(labels_dict.get(seg[-1], 0))

    return seqs, lbls
