# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Builds an image embedding index from images/ using MobileNetV2, 
# paired with angles from data.json (theta, phi, roll).
# Saves:
#   - emb_train.npy        (float32, shape: [N, 1280])
#   - theta_train.npy      (float32, shape: [N])
#   - phi_train.npy        (float32, shape: [N])
#   - roll_train.npy       (float32, shape: [N])
#   - paths_train.npy      (object,  shape: [N])
#   - index_summary.csv    (for quick inspection)
# """

# import json
# import sys
# from pathlib import Path

# import numpy as np
# import pandas as pd

# from tensorflow.keras.utils import load_img, img_to_array
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from tensorflow.keras.models import Model

# IMG_SIZE = (224, 224)

# def load_metadata(json_path: Path) -> dict:
#     """Load angles metadata (theta, phi, roll) from data.json."""
#     if not json_path.exists():
#         raise FileNotFoundError(f"Missing JSON file: {json_path}")
#     with open(json_path, "r", encoding="utf-8") as f:
#         meta = json.load(f)
#     return meta

# def build_backbone() -> Model:
#     """MobileNetV2 backbone (ImageNet), global average pooled embeddings (1280-d)."""
#     base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
#     return base

# def load_and_embed(img_path: Path, model: Model) -> np.ndarray:
#     """Load image, preprocess, and extract a 1280-d embedding."""
#     # Load & resize
#     img = load_img(img_path, target_size=IMG_SIZE)  # RGB
#     arr = img_to_array(img)                         # (H,W,3), float32
#     arr = np.expand_dims(arr, axis=0)               # (1,H,W,3)
#     arr = preprocess_input(arr)                     # MobileNetV2 preprocessing
#     emb = model.predict(arr, verbose=0)             # (1,1280)
#     return emb.squeeze(0).astype(np.float32)        # (1280,)

# def main():
#     root = Path(__file__).resolve().parent
#     images_dir = root / "images"
#     json_path = images_dir / "data.json"

#     print("[INFO] Loading metadata...")
#     meta = load_metadata(json_path)

#     # Collect image file list that appear in the JSON and actually exist on disk
#     print("[INFO] Scanning images directory...")
#     all_items = []
#     for name in sorted(meta.keys()):
#         p = images_dir / name
#         if p.exists() and p.is_file():
#             angles = meta[name]
#             all_items.append((p, angles))
#         else:
#             print(f"[WARN] Skipping missing file: {p}")

#     if not all_items:
#         print("[ERROR] No valid images found that match data.json entries.")
#         sys.exit(1)

#     print(f"[INFO] Found {len(all_items)} images. Building backbone...")
#     backbone = build_backbone()

#     embeddings = []
#     thetas = []
#     phis = []
#     rolls = []
#     paths = []

#     print("[INFO] Extracting embeddings...")
#     for idx, (img_path, angles) in enumerate(all_items, start=1):
#         try:
#             emb = load_and_embed(img_path, backbone)
#             embeddings.append(emb)
#             thetas.append(float(angles.get("theta", 0.0)))
#             phis.append(float(angles.get("phi", 0.0)))
#             rolls.append(float(angles.get("roll", 0.0)))
#             paths.append(str(img_path))
#         except Exception as e:
#             print(f"[ERROR] Failed on {img_path}: {e}")

#         if idx % 10 == 0:
#             print(f"[INFO] Processed {idx}/{len(all_items)} images...")

#     if not embeddings:
#         print("[ERROR] No embeddings were created. Aborting.")
#         sys.exit(1)

#     emb_np = np.stack(embeddings, axis=0)  # (N,1280)
#     theta_np = np.array(thetas, dtype=np.float32)
#     phi_np = np.array(phis, dtype=np.float32)
#     roll_np = np.array(rolls, dtype=np.float32)
#     paths_np = np.array(paths, dtype=object)

#     # Save artifacts in project root
#     np.save(root / "emb_train.npy", emb_np)
#     np.save(root / "theta_train.npy", theta_np)
#     np.save(root / "phi_train.npy", phi_np)
#     np.save(root / "roll_train.npy", roll_np)
#     np.save(root / "paths_train.npy", paths_np)

#     # Optional quick CSV for sanity check
#     df = pd.DataFrame({
#         "path": paths_np,
#         "theta": theta_np,
#         "phi": phi_np,
#         "roll": roll_np
#     })
#     df.to_csv(root / "index_summary.csv", index=False)

#     print("[OK] Saved:")
#     print(" - emb_train.npy      (embeddings)")
#     print(" - theta_train.npy    (theta)")
#     print(" - phi_train.npy      (phi)")
#     print(" - roll_train.npy     (roll)")
#     print(" - paths_train.npy    (image paths)")
#     print(" - index_summary.csv  (overview)")
#     print(f"[DONE] Total embeddings: {emb_np.shape[0]} | Dim: {emb_np.shape[1]}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds an image embedding index from images/ using MobileNetV2,
paired with angles from data.json (theta, phi, roll).

Saves in project root:
  - emb_train.npy        (float32, shape: [N, 1280])
  - theta_train.npy      (float32, shape: [N])
  - phi_train.npy        (float32, shape: [N])
  - roll_train.npy       (float32, shape: [N])
  - paths_train.npy      (object,  shape: [N])
  - index_summary.csv    (overview of paths and angles)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model

IMG_SIZE = (224, 224)


def load_metadata(json_path: Path) -> dict:
    """Load angles metadata (theta, phi, roll) from data.json."""
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def build_backbone() -> Model:
    """MobileNetV2 backbone (ImageNet), global average pooled embeddings (1280-d)."""
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    return base


def load_and_preprocess(img_path: Path, crop_frac: float = 0.85) -> np.ndarray:
    """
    Load image, center-crop by crop_frac, resize to 224x224, preprocess for MobileNetV2.
    Returns a batch array of shape (1, 224, 224, 3).
    """
    img = load_img(img_path)  # PIL.Image
    w, h = img.size
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    left = (w - cw) // 2
    top = (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))   # center crop
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = img_to_array(img)                            # (H, W, 3), float32
    arr = np.expand_dims(arr, axis=0)                  # (1, H, W, 3)
    arr = preprocess_input(arr)
    return arr


def load_and_embed(img_path: Path, model: Model) -> np.ndarray:
    """Load image, preprocess, and extract a 1280-d embedding."""
    arr = load_and_preprocess(img_path, crop_frac=0.85)
    emb = model.predict(arr, verbose=0)                # (1, 1280)
    return emb.squeeze(0).astype(np.float32)           # (1280,)


def main():
    root = Path(__file__).resolve().parent
    images_dir = root / "images"
    json_path = images_dir / "data.json"

    print("[INFO] Loading metadata...")
    meta = load_metadata(json_path)

    print("[INFO] Scanning images directory...")
    all_items = []
    for name in sorted(meta.keys()):
        p = images_dir / name
        if p.exists() and p.is_file():
            angles = meta[name]
            all_items.append((p, angles))
        else:
            print(f"[WARN] Skipping missing file: {p}")

    if not all_items:
        print("[ERROR] No valid images found that match data.json entries.")
        sys.exit(1)

    print(f"[INFO] Found {len(all_items)} images. Building backbone...")
    backbone = build_backbone()

    embeddings = []
    thetas = []
    phis = []
    rolls = []
    paths = []

    print("[INFO] Extracting embeddings...")
    for idx, (img_path, angles) in enumerate(all_items, start=1):
        try:
            emb = load_and_embed(img_path, backbone)
            embeddings.append(emb)
            thetas.append(float(angles.get("theta", 0.0)))
            phis.append(float(angles.get("phi", 0.0)))
            rolls.append(float(angles.get("roll", 0.0)))
            paths.append(str(img_path))
        except Exception as e:
            print(f"[ERROR] Failed on {img_path}: {e}")

        if idx % 10 == 0 or idx == len(all_items):
            print(f"[INFO] Processed {idx}/{len(all_items)} images...")

    if not embeddings:
        print("[ERROR] No embeddings were created. Aborting.")
        sys.exit(1)

    emb_np = np.stack(embeddings, axis=0)  # (N, 1280)
    theta_np = np.array(thetas, dtype=np.float32)
    phi_np = np.array(phis, dtype=np.float32)
    roll_np = np.array(rolls, dtype=np.float32)
    paths_np = np.array(paths, dtype=object)

    # Save artifacts in project root
    np.save(root / "emb_train.npy", emb_np)
    np.save(root / "theta_train.npy", theta_np)
    np.save(root / "phi_train.npy", phi_np)
    np.save(root / "roll_train.npy", roll_np)
    np.save(root / "paths_train.npy", paths_np)

    # Quick CSV for sanity check
    df = pd.DataFrame({
        "path": paths_np,
        "theta": theta_np,
        "phi": phi_np,
        "roll": roll_np
    })
    df.to_csv(root / "index_summary.csv", index=False)

    print("[OK] Saved:")
    print(" - emb_train.npy      (embeddings)")
    print(" - theta_train.npy    (theta)")
    print(" - phi_train.npy      (phi)")
    print(" - roll_train.npy     (roll)")
    print(" - paths_train.npy    (image paths)")
    print(" - index_summary.csv  (overview)")
    print(f"[DONE] Total embeddings: {emb_np.shape[0]} | Dim: {emb_np.shape[1]}")


if __name__ == "__main__":
    main()
