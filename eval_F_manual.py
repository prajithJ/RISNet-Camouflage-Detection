import os
import cv2
import numpy as np
from tqdm import tqdm


gt_root = "/DATA/risnet_work/data/ACOD-12K/Test/GT/"
pred_root = "/DATA/risnet_work/code/RISNet/results/ACOD-12K/"

beta2 = 0.3  # beta^2, standard in SOD / camouflage


def compute_f_measure(pred, gt, beta2):
    """
    pred, gt: binary numpy arrays {0,1}
    """
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    return f


# Collect F scores over thresholds
F_scores = []

thresholds = np.linspace(0, 1, 256)

for t in tqdm(thresholds, desc="Threshold sweep"):
    f_list = []

    for name in os.listdir(gt_root):
        gt = cv2.imread(os.path.join(gt_root, name), 0)
        pred = cv2.imread(os.path.join(pred_root, name), 0)

        if gt is None or pred is None:
            continue

        gt = (gt > 128).astype(np.uint8)
        pred_norm = pred.astype(np.float32) / 255.0
        pred_bin = (pred_norm >= t).astype(np.uint8)

        f = compute_f_measure(pred_bin, gt, beta2)
        f_list.append(f)

    F_scores.append(np.mean(f_list))

F_scores = np.array(F_scores)

print("\n===== Manual F-measure Results =====")
print("F-measure (max):", float(F_scores.max()))
