# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# KNN-based retrieval for camera pose (theta, phi, roll) using MobileNetV2 embeddings.

# Usage examples:
#   # Evaluate with softmax weighting, k=5, temperature=0.07, and min similarity 0.55
#   python query_knn.py --eval --k 5 --weighting softmax --tau 0.07 --min_sim 0.55

#   # Single image prediction
#   python query_knn.py --image images_test/image_01.jpg --k 5 --weighting softmax --tau 0.07 --min_sim 0.55
# """

# import json
# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd

# from sklearn.metrics.pairwise import cosine_similarity
# from tensorflow.keras.utils import load_img, img_to_array
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# IMG_SIZE = (224, 224)

# def build_backbone():
#     return MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

# def embed_image(img_path: Path, backbone) -> np.ndarray:
#     img = load_img(img_path, target_size=IMG_SIZE)
#     arr = img_to_array(img)
#     arr = np.expand_dims(arr, axis=0)
#     arr = preprocess_input(arr)
#     emb = backbone.predict(arr, verbose=0)
#     return emb.squeeze(0).astype(np.float32)

# def circ_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
#     ang = np.deg2rad(angles_deg)
#     w = weights.astype(np.float64)
#     s = np.sum(w * np.sin(ang))
#     c = np.sum(w * np.cos(ang))
#     mean = np.arctan2(s, c)
#     mean_deg = np.rad2deg(mean)
#     return float((mean_deg + 360.0) % 360.0)

# def circ_diff_abs_deg(a_deg: float, b_deg: float) -> float:
#     d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
#     return float(abs(d))

# # ---------- new: weighting helpers ----------
# def softmax_weights(sims: np.ndarray, tau: float = 0.07) -> np.ndarray:
#     """Softmax over similarities with temperature tau (smaller tau -> sharper)."""
#     z = (sims - sims.max()) / max(tau, 1e-8)  # stabilize
#     w = np.exp(z)
#     s = w.sum()
#     return w / s if s > 1e-12 else np.ones_like(sims) / len(sims)

# def linear_weights(sims: np.ndarray) -> np.ndarray:
#     """Linear weights: shift-min then normalize."""
#     w = sims - sims.min()
#     s = w.sum()
#     return w / s if s > 1e-12 else np.ones_like(sims) / len(sims)
# # --------------------------------------------

# def load_train_arrays(root: Path):
#     emb = np.load(root / "emb_train.npy")
#     theta = np.load(root / "theta_train.npy")
#     phi = np.load(root / "phi_train.npy")
#     roll = np.load(root / "roll_train.npy")
#     paths = np.load(root / "paths_train.npy", allow_pickle=True)
#     return emb, theta, phi, roll, paths

# def knn_predict(query_emb: np.ndarray,
#                 emb_train: np.ndarray,
#                 theta: np.ndarray, phi: np.ndarray, roll: np.ndarray,
#                 k: int = 5,
#                 weighting: str = "softmax",
#                 tau: float = 0.07,
#                 min_sim: float = 0.55):
#     """
#     Compute cosine similarity to all train embeddings, select top-k,
#     optionally filter by min_sim, then aggregate using chosen weighting.
#     """
#     sims = cosine_similarity(query_emb[None, :], emb_train)[0]  # (N,)
#     idx_topk = np.argsort(-sims)[:k]
#     idx = idx_topk[sims[idx_topk] >= min_sim]  # keep only sufficiently similar neighbors

#     # Fallback: if nothing passes the threshold, use the best single neighbor
#     if idx.size == 0:
#         idx = idx_topk[:1]

#     top_sims = sims[idx]

#     if weighting == "softmax":
#         weights = softmax_weights(top_sims, tau=tau)
#     else:
#         weights = linear_weights(top_sims)

#     phi_pred = circ_mean_deg(phi[idx], weights)
#     theta_pred = circ_mean_deg(theta[idx], weights)
#     roll_pred = circ_mean_deg(roll[idx], weights)

#     return {
#         "indices": idx,
#         "similarities": top_sims,
#         "weights": weights,
#         "phi_pred": phi_pred,
#         "theta_pred": theta_pred,
#         "roll_pred": roll_pred
#     }

# def evaluate_all(root: Path, k: int = 5, weighting: str = "softmax", tau: float = 0.07, min_sim: float = 0.55):
#     test_dir = root / "images_test"
#     meta_path = test_dir / "data_test.json"
#     if not meta_path.exists():
#         print("[ERROR] images_test/data_test.json not found.")
#         return None
#     with open(meta_path, "r", encoding="utf-8") as f:
#         meta = json.load(f)

#     emb_train, theta, phi, roll, paths = load_train_arrays(root)
#     backbone = build_backbone()

#     rows = []
#     for name, gt in sorted(meta.items()):
#         img_path = test_dir / name
#         if not img_path.exists():
#             print(f"[WARN] Missing test image: {img_path}")
#             continue

#         q_emb = embed_image(img_path, backbone)
#         out = knn_predict(q_emb, emb_train, theta, phi, roll, k=k, weighting=weighting, tau=tau, min_sim=min_sim)

#         phi_gt = float(gt.get("phi", 0.0))
#         theta_gt = float(gt.get("theta", 0.0))
#         roll_gt = float(gt.get("roll", 0.0))

#         phi_err = circ_diff_abs_deg(out["phi_pred"], phi_gt)
#         theta_err = circ_diff_abs_deg(out["theta_pred"], theta_gt)
#         roll_err = circ_diff_abs_deg(out["roll_pred"], roll_gt)

#         rows.append({
#             "image": str(img_path),
#             "phi_gt": phi_gt, "phi_pred": out["phi_pred"], "phi_err": phi_err,
#             "theta_gt": theta_gt, "theta_pred": out["theta_pred"], "theta_err": theta_err,
#             "roll_gt": roll_gt, "roll_pred": out["roll_pred"], "roll_err": roll_err
#         })

#         used = ", ".join([f"{paths[i]}(sim={s:.3f},w={w:.3f})" for i, s, w in zip(out["indices"], out["similarities"], out["weights"])])
#         print(f"[TEST] {name} | phi_gt={phi_gt:.1f} phi_pred={out['phi_pred']:.1f} phi_err={phi_err:.1f} | used: {used}")

#     if not rows:
#         print("[ERROR] No test rows to evaluate.")
#         return None

#     df = pd.DataFrame(rows)
#     print("\n[SUMMARY] MAE (mean absolute error):")
#     print(f" - phi_mae   = {df['phi_err'].mean():.2f} deg")
#     print(f" - theta_mae = {df['theta_err'].mean():.2f} deg")
#     print(f" - roll_mae  = {df['roll_err'].mean():.2f} deg")
#     df.to_csv(root / "eval_results.csv", index=False)
#     print("[OK] Saved eval_results.csv")
#     return df

# def single_image_predict(root: Path, image_path: Path, k: int = 5, weighting: str = "softmax", tau: float = 0.07, min_sim: float = 0.55):
#     emb_train, theta, phi, roll, paths = load_train_arrays(root)
#     backbone = build_backbone()
#     q_emb = embed_image(image_path, backbone)
#     out = knn_predict(q_emb, emb_train, theta, phi, roll, k=k, weighting=weighting, tau=tau, min_sim=min_sim)

#     print(f"[PRED] image: {image_path}")
#     print(f"[PRED] theta={out['theta_pred']:.2f} | phi={out['phi_pred']:.2f} | roll={out['roll_pred']:.2f}")
#     print("[PRED] used neighbors:")
#     for rank, (i, s, w) in enumerate(zip(out["indices"], out["similarities"], out["weights"]), start=1):
#         print(f"  #{rank:02d}  sim={s:.4f}  w={w:.3f}  path={paths[i]}  "
#               f"theta={theta[i]:.1f} phi={phi[i]:.1f} roll={roll[i]:.1f}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--eval", action="store_true", help="Evaluate on images_test/data_test.json")
#     parser.add_argument("--image", type=str, help="Path to a single image to predict")
#     parser.add_argument("--k", type=int, default=5, help="Num neighbors")
#     parser.add_argument("--weighting", type=str, default="softmax", choices=["softmax", "linear"], help="Neighbor weighting method")
#     parser.add_argument("--tau", type=float, default=0.07, help="Softmax temperature")
#     parser.add_argument("--min_sim", type=float, default=0.55, help="Minimum cosine similarity to keep a neighbor")
#     args = parser.parse_args()

#     root = Path(__file__).resolve().parent

#     if args.eval:
#         evaluate_all(root, k=args.k, weighting=args.weighting, tau=args.tau, min_sim=args.min_sim)
#     elif args.image:
#         img_path = Path(args.image)
#         if not img_path.exists():
#             print(f"[ERROR] Image not found: {img_path}")
#             return
#         single_image_predict(root, img_path, k=args.k, weighting=args.weighting, tau=args.tau, min_sim=args.min_sim)
#     else:
#         print("Nothing to do. Use --eval or --image.")
#         print("Examples:")
#         print("  python query_knn.py --eval --k 5 --weighting softmax --tau 0.07 --min_sim 0.55")
#         print("  python query_knn.py --image images_test/image_01.jpg --k 5 --weighting softmax --tau 0.07 --min_sim 0.55")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KNN-based retrieval for camera pose (theta, phi, roll) using MobileNetV2 embeddings.

Usage examples:
  # Evaluate test set with tuned parameters:
  python query_knn.py --eval --k 3 --weighting softmax --tau 0.07 --min_sim 0.55 --angle_window 30

  # Predict a single image:
  python query_knn.py --image images_test/image_01.jpg --k 3 --weighting softmax --tau 0.07 --min_sim 0.55 --angle_window 30
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

IMG_SIZE = (224, 224)


def build_backbone():
    """MobileNetV2 backbone (ImageNet), global average pooled embeddings (1280-d)."""
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))


def preprocess_for_backbone(img_path: Path, crop_frac: float = 0.85) -> np.ndarray:
    """
    Center-crop by crop_frac, resize to 224x224, preprocess for MobileNetV2.
    Returns a batch array of shape (1, 224, 224, 3).
    """
    img = load_img(img_path)  # PIL.Image
    w, h = img.size
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    left = (w - cw) // 2
    top = (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def embed_image(img_path: Path, backbone) -> np.ndarray:
    """Return a (1280,) embedding for a single image."""
    arr = preprocess_for_backbone(img_path, crop_frac=0.85)
    emb = backbone.predict(arr, verbose=0)
    return emb.squeeze(0).astype(np.float32)


def circ_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean (degrees)."""
    ang = np.deg2rad(angles_deg)
    w = weights.astype(np.float64)
    s = np.sum(w * np.sin(ang))
    c = np.sum(w * np.cos(ang))
    mean = np.arctan2(s, c)
    mean_deg = np.rad2deg(mean)
    return float((mean_deg + 360.0) % 360.0)


def circ_diff_abs_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute circular difference (degrees)."""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return float(abs(d))


# ---------- weighting helpers ----------
def softmax_weights(sims: np.ndarray, tau: float = 0.07) -> np.ndarray:
    """Softmax over similarities with temperature tau (smaller tau -> sharper)."""
    z = (sims - sims.max()) / max(tau, 1e-8)  # stabilize
    w = np.exp(z)
    s = w.sum()
    return w / s if s > 1e-12 else np.ones_like(sims) / len(sims)


def linear_weights(sims: np.ndarray) -> np.ndarray:
    """Linear weights: shift-min then normalize."""
    w = sims - sims.min()
    s = w.sum()
    return w / s if s > 1e-12 else np.ones_like(sims) / len(sims)
# ---------------------------------------


def load_train_arrays(root: Path):
    emb = np.load(root / "emb_train.npy")
    theta = np.load(root / "theta_train.npy")
    phi = np.load(root / "phi_train.npy")
    roll = np.load(root / "roll_train.npy")
    paths = np.load(root / "paths_train.npy", allow_pickle=True)
    return emb, theta, phi, roll, paths


def knn_predict(query_emb: np.ndarray,
                emb_train: np.ndarray,
                theta: np.ndarray, phi: np.ndarray, roll: np.ndarray,
                k: int = 5,
                weighting: str = "softmax",
                tau: float = 0.07,
                min_sim: float = 0.55,
                angle_window: float = 30.0):
    """
    Compute cosine similarity to all train embeddings, select top-k,
    filter by min_sim, aggregate with selected weighting,
    then apply a consistency filter around first-pass phi prediction.
    """
    sims = cosine_similarity(query_emb[None, :], emb_train)[0]  # (N,)
    idx_topk = np.argsort(-sims)[:k]
    idx = idx_topk[sims[idx_topk] >= min_sim]

    # Fallback: if nothing passes the threshold, use the best single neighbor
    if idx.size == 0:
        idx = idx_topk[:1]

    top_sims = sims[idx]
    if weighting == "softmax":
        weights = softmax_weights(top_sims, tau=tau)
    else:
        weights = linear_weights(top_sims)

    # first-pass prediction
    phi_pred = circ_mean_deg(phi[idx], weights)
    theta_pred = circ_mean_deg(theta[idx], weights)
    roll_pred = circ_mean_deg(roll[idx], weights)

    # consistency filter around phi_pred (keep neighbors within angle_window degrees)
    diffs = np.array([circ_diff_abs_deg(a, phi_pred) for a in phi[idx]])
    keep = diffs <= angle_window
    if keep.sum() >= 2 and keep.sum() < len(idx):
        idx = idx[keep]
        top_sims = top_sims[keep]
        if weighting == "softmax":
            weights = softmax_weights(top_sims, tau=tau)
        else:
            weights = linear_weights(top_sims)
        phi_pred = circ_mean_deg(phi[idx], weights)
        theta_pred = circ_mean_deg(theta[idx], weights)
        roll_pred = circ_mean_deg(roll[idx], weights)

    return {
        "indices": idx,
        "similarities": top_sims,
        "weights": weights,
        "phi_pred": phi_pred,
        "theta_pred": theta_pred,
        "roll_pred": roll_pred
    }


def evaluate_all(root: Path,
                 k: int = 5,
                 weighting: str = "softmax",
                 tau: float = 0.07,
                 min_sim: float = 0.55,
                 angle_window: float = 30.0):
    """Evaluate on images_test/ using data_test.json; prints MAE and saves eval_results.csv."""
    test_dir = root / "images_test"
    meta_path = test_dir / "data_test.json"
    if not meta_path.exists():
        print("[ERROR] images_test/data_test.json not found.")
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    emb_train, theta, phi, roll, paths = load_train_arrays(root)
    backbone = build_backbone()

    rows = []
    for name, gt in sorted(meta.items()):
        img_path = test_dir / name
        if not img_path.exists():
            print(f"[WARN] Missing test image: {img_path}")
            continue

        q_emb = embed_image(img_path, backbone)
        out = knn_predict(
            q_emb, emb_train, theta, phi, roll,
            k=k, weighting=weighting, tau=tau, min_sim=min_sim, angle_window=angle_window
        )

        phi_gt = float(gt.get("phi", 0.0))
        theta_gt = float(gt.get("theta", 0.0))
        roll_gt = float(gt.get("roll", 0.0))

        phi_err = circ_diff_abs_deg(out["phi_pred"], phi_gt)
        theta_err = circ_diff_abs_deg(out["theta_pred"], theta_gt)
        roll_err = circ_diff_abs_deg(out["roll_pred"], roll_gt)

        rows.append({
            "image": str(img_path),
            "phi_gt": phi_gt, "phi_pred": out["phi_pred"], "phi_err": phi_err,
            "theta_gt": theta_gt, "theta_pred": out["theta_pred"], "theta_err": theta_err,
            "roll_gt": roll_gt, "roll_pred": out["roll_pred"], "roll_err": roll_err
        })

        used = ", ".join([f"{paths[i]}(sim={s:.3f},w={w:.3f})"
                          for i, s, w in zip(out["indices"], out["similarities"], out["weights"])])
        print(f"[TEST] {name} | phi_gt={phi_gt:.1f} phi_pred={out['phi_pred']:.1f} phi_err={phi_err:.1f} | used: {used}")

    if not rows:
        print("[ERROR] No test rows to evaluate.")
        return None

    df = pd.DataFrame(rows)
    print("\n[SUMMARY] MAE (mean absolute error):")
    print(f" - phi_mae   = {df['phi_err'].mean():.2f} deg")
    print(f" - theta_mae = {df['theta_err'].mean():.2f} deg")
    print(f" - roll_mae  = {df['roll_err'].mean():.2f} deg")
    df.to_csv(root / "eval_results.csv", index=False)
    print("[OK] Saved eval_results.csv")
    return df


def single_image_predict(root: Path,
                         image_path: Path,
                         k: int = 5,
                         weighting: str = "softmax",
                         tau: float = 0.07,
                         min_sim: float = 0.55,
                         angle_window: float = 30.0):
    """Single image prediction with neighbor breakdown."""
    emb_train, theta, phi, roll, paths = load_train_arrays(root)
    backbone = build_backbone()
    q_emb = embed_image(image_path, backbone)
    out = knn_predict(
        q_emb, emb_train, theta, phi, roll,
        k=k, weighting=weighting, tau=tau, min_sim=min_sim, angle_window=angle_window
    )

    print(f"[PRED] image: {image_path}")
    print(f"[PRED] theta={out['theta_pred']:.2f} | phi={out['phi_pred']:.2f} | roll={out['roll_pred']:.2f}")
    print("[PRED] used neighbors:")
    for rank, (i, s, w) in enumerate(zip(out["indices"], out["similarities"], out["weights"]), start=1):
        print(f"  #{rank:02d}  sim={s:.4f}  w={w:.3f}  path={paths[i]}  "
              f"theta={theta[i]:.1f} phi={phi[i]:.1f} roll={roll[i]:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate on images_test/data_test.json")
    parser.add_argument("--image", type=str, help="Path to a single image to predict")
    parser.add_argument("--k", type=int, default=3, help="Num neighbors")
    parser.add_argument("--weighting", type=str, default="softmax",
                        choices=["softmax", "linear"], help="Neighbor weighting method")
    parser.add_argument("--tau", type=float, default=0.07, help="Softmax temperature")
    parser.add_argument("--min_sim", type=float, default=0.55, help="Minimum cosine similarity to keep a neighbor")
    parser.add_argument("--angle_window", type=float, default=30.0,
                        help="Max circular deviation (deg) to keep neighbors around first-pass phi")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    if args.eval:
        evaluate_all(root, k=args.k, weighting=args.weighting, tau=args.tau,
                     min_sim=args.min_sim, angle_window=args.angle_window)
    elif args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[ERROR] Image not found: {img_path}")
            return
        single_image_predict(root, img_path, k=args.k, weighting=args.weighting, tau=args.tau,
                             min_sim=args.min_sim, angle_window=args.angle_window)
    else:
        print("Nothing to do. Use --eval or --image.")
        print("Examples:")
        print("  python query_knn.py --eval --k 3 --weighting softmax --tau 0.07 --min_sim 0.55 --angle_window 30")
        print("  python query_knn.py --image images_test/image_01.jpg --k 3 --weighting softmax --tau 0.07 --min_sim 0.55 --angle_window 30")


if __name__ == "__main__":
    main()
