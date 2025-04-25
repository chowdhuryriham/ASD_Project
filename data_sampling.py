# data_sampling.py

import numpy as np
from sklearn.utils import resample

def augment_data(X, noise_std=0.01, num_augments=2):
    """
    Augment each sequence in X by adding Gaussian noise.
    
    Parameters:
      X: numpy array of shape (n_samples, seq_length, num_features)
      noise_std: standard deviation for Gaussian noise
      num_augments: number of augmented copies per original sample
      
    Returns:
      A numpy array containing all augmented samples.
    """
    augmented = []
    for seq in X:
        for _ in range(num_augments):
            noise = np.random.normal(0.0, noise_std, size=seq.shape)
            augmented.append(seq + noise)
    return np.array(augmented, dtype=X.dtype)


def oversample_data(X, y, augment=False, noise_std=0.01, num_augments=1):
    """
    Oversample minority classes to balance the dataset, with optional augmentation.
    
    Parameters:
      X : array-like of shape (n_samples, ...), feature sequences
      y : array-like of shape (n_samples,), integer class labels
      augment: whether to augment minority samples after resampling
      noise_std: Gaussian noise level for augmentation
      num_augments: number of augmented copies per minority sample
    
    Returns:
      X_over, y_over: oversampled (and augmented) arrays
    """
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_list, y_list = [], []

    for cls, cnt in zip(unique, counts):
        idx = np.where(y == cls)[0]
        X_cls, y_cls = X[idx], y[idx]

        # 1) Resample up to max_count
        if cnt < max_count:
            X_res, y_res = resample(
                X_cls, y_cls,
                replace=True,
                n_samples=max_count,
                random_state=42
            )
        else:
            X_res, y_res = X_cls, y_cls

        # 2) Optionally augment the original minority examples
        if augment and cnt < max_count:
            X_aug = augment_data(X_cls, noise_std=noise_std, num_augments=num_augments)
            y_aug = np.full(X_aug.shape[0], cls, dtype=y.dtype)

            # Combine and, if too many, randomly trim
            X_comb = np.vstack([X_res, X_aug])
            y_comb = np.concatenate([y_res, y_aug], axis=0)
            if X_comb.shape[0] > max_count:
                sel = np.random.choice(X_comb.shape[0], max_count, replace=False)
                X_res, y_res = X_comb[sel], y_comb[sel]
            else:
                X_res, y_res = X_comb, y_comb

        X_list.append(X_res)
        y_list.append(y_res)

    X_over = np.vstack(X_list)
    y_over = np.concatenate(y_list, axis=0)
    return X_over, y_over
