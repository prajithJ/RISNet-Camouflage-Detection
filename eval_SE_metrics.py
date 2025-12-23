import os
import cv2
import numpy as np
from tqdm import tqdm

from py_sod_metrics import Smeasure, Emeasure


gt_root = "/DATA/risnet_work/data/ACOD-12K/Test/GT/"
pred_root = "/DATA/risnet_work/code/RISNet/results/ACOD-12K/"


sm = Smeasure()
em = Emeasure()

for name in tqdm(os.listdir(gt_root)):
    gt_path = os.path.join(gt_root, name)
    pred_path = os.path.join(pred_root, name)

    gt = cv2.imread(gt_path, 0)
    pred = cv2.imread(pred_path, 0)

    if gt is None or pred is None:
        continue

    # ---- IMPORTANT: GT must be binary ----
    gt = (gt > 128).astype(np.uint8)

    # Prediction stays continuous for S & E
    pred_norm = pred.astype(np.float32) / 255.0

    # Update metrics
    sm.step(pred=pred_norm, gt=gt)
    em.step(pred=pred_norm, gt=gt)


# ---- Results (THIS IS THE CORRECT ACCESS) ----
sm_res = sm.get_results()
em_res = em.get_results()

S_score = sm_res["sm"]
E_curve = em_res["em"]["curve"]

print("\n===== S & E Evaluation Results =====")
print("S-measure       :", float(S_score))
print("E-measure mean  :", float(E_curve.mean()))
print("E-measure max   :", float(E_curve.max()))

