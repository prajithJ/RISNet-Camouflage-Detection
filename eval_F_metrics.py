import os
import cv2
import numpy as np
from tqdm import tqdm

from py_sod_metrics import FmeasureV2


gt_root = "/DATA/risnet_work/data/ACOD-12K/Test/GT/"
pred_root = "/DATA/risnet_work/code/RISNet/results/ACOD-12K/"


# -----------------------------
# Initialize F-measure V2
# -----------------------------
fm = FmeasureV2()

# ---- ADD METRIC HANDLER (CRITICAL STEP) ----
fm.add_handler("max")        # max F-measure over thresholds
fm.add_handler("adaptive")   # adaptive F-measure (optional but standard)


# -----------------------------
# Evaluation loop
# -----------------------------
for name in tqdm(os.listdir(gt_root)):
    gt_path = os.path.join(gt_root, name)
    pred_path = os.path.join(pred_root, name)

    gt = cv2.imread(gt_path, 0)
    pred = cv2.imread(pred_path, 0)

    if gt is None or pred is None:
        continue

    # GT must be binary
    gt = (gt > 128).astype(np.uint8)

    # Prediction must be binary for F-measure
    pred_norm = pred.astype(np.float32) / 255.0
    pred_bin = (pred_norm > 0.5).astype(np.uint8)

    fm.step(pred=pred_bin, gt=gt)


# -----------------------------
# Results
# -----------------------------
fm_res = fm.get_results()["fm"]

F_max = fm_res["max"]
F_adaptive = fm_res["adaptive"]

print("\n===== F-measure Evaluation Results =====")
print("F-measure (max)      :", float(F_max))
print("F-measure (adaptive) :", float(F_adaptive))

