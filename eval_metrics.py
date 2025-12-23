import os
import cv2
import numpy as np
from tqdm import tqdm

from py_sod_metrics import Smeasure, Emeasure


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = img.astype(np.float32)
    if img.max() > 1:
        img /= 255.0
    return img


def evaluate_SE(pred_dir, gt_dir):

    sm = Smeasure(alpha=0.5)
    em = Emeasure()

    names = [n for n in os.listdir(gt_dir) if n.endswith(".png")]
    names.sort()

    for name in tqdm(names):
        gt_path = os.path.join(gt_dir, name)
        pred_path = os.path.join(pred_dir, name)

        if not os.path.exists(pred_path):
            continue

        gt = load_gray(gt_path)
        pred = load_gray(pred_path)

        assert gt.shape == pred.shape

        sm.step(pred=pred, gt=gt)
        em.step(pred=pred, gt=gt)

    # ---- S-measure (stable) ----
    S_score = sm.get_results()["sm"]

    # ---- E-measure (version-safe) ----
    E_curve = em.get_results()["em"]   # ndarray in your version
    E_mean = float(np.mean(E_curve))
    E_max = float(np.max(E_curve))

    return {
        "S_measure": S_score,
        "E_mean": E_mean,
        "E_max": E_max
    }


if __name__ == "__main__":

    pred_dir = "./results/ACOD-12K"
    gt_dir = "/DATA/risnet_work/data/ACOD-12K/Test/GT"

    metrics = evaluate_SE(pred_dir, gt_dir)

    print("\n===== S & E Evaluation Results =====")
    print(f"S-measure        : {metrics['S_measure']:.4f}")
    print(f"E-measure (mean) : {metrics['E_mean']:.4f}")
    print(f"E-measure (max)  : {metrics['E_max']:.4f}")


