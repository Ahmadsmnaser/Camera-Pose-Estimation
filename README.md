# Simple Camera Pose Estimation (Aseel-style)

## What we did
- Used **MobileNetV2 (pretrained)** to extract 1280-D embeddings from images (center-crop 0.85 → 224×224).
- Built an **index** from 20 labeled images (`data.json` with theta/phi/roll).
- For a query image: computed **cosine similarity**, took **top-k (k=3)**, applied **softmax weighting** (τ=0.05) + **min similarity** (0.60) + **angular consistency** (±35°), then predicted **φ** via **weighted circular mean**.
- Evaluated on 10 test images → **MAE(φ) ≈ 6.70°**.

## Run
```bash
py -3.12 build_index.py
py -3.12 query_knn.py --eval --k 3 --weighting softmax --tau 0.05 --min_sim 0.60 --angle_window 35
# Single image:
py -3.12 query_knn.py --image images_test/image_01.jpg --k 3 --weighting softmax --tau 0.05 --min_sim 0.60 --angle_window 35
