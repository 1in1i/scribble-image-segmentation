import os
import glob
import random
from collections import deque

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader

# Assuming data_loader.py exists and contains ScribbleSegmentationDataset and get_dataloaders
from data_loader import ScribbleSegmentationDataset, get_dataloaders

# ---------------------------
# K-Means Segmentation
# ---------------------------
def kmeans_segmentation(image_np: np.ndarray, scribble_np: np.ndarray, k: int = 16) -> np.ndarray:
    """
    Performs image segmentation using a seeded K-Means algorithm.

    Args:
        image_np: The input image as a NumPy array (H, W, 3).
        scribble_np: The scribble mask as a NumPy array (H, W).
        k: The number of clusters for K-Means.

    Returns:
        A binary segmentation mask (H, W) as a NumPy array.
    """
    H, W, C = image_np.shape

    # 1. Reshape the image to be a list of pixels (N_pixels, 3)
    pixel_data = image_np.reshape((-1, C)).astype(np.float32)

    # 2. Run K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 3. Identify the foreground clusters using the scribbles
    scribble_flat = scribble_np.flatten()
    fg_indices = (scribble_flat == 1)
    
    # Find the unique cluster labels that the foreground scribbles fall into
    if np.any(fg_indices):
        fg_clusters = np.unique(labels[fg_indices])
    else:
        fg_clusters = []

    # 4. Create the output mask by labeling all pixels belonging to foreground clusters
    output_mask_flat = np.zeros_like(labels, dtype=np.uint8)
    for cluster_id in fg_clusters:
        output_mask_flat[labels == cluster_id] = 1

    # 5. Reshape the mask back to the original image dimensions
    output_mask = output_mask_flat.reshape((H, W))

    return output_mask

def enforce_scribble_constraints(pred_bin: np.ndarray, scribble_np: np.ndarray, keep_component=True) -> np.ndarray:
    out = pred_bin.copy()
    out[scribble_np == 0] = 0
    out[scribble_np == 1] = 1
    if keep_component:
        seeds = np.argwhere(scribble_np == 1)
        if seeds.size>0:
            H,W = out.shape
            vis = np.zeros_like(out, dtype=np.uint8)
            q = deque([(int(r),int(c)) for r,c in seeds if out[int(r),int(c)]==1])
            for r,c in q:
                vis[r,c]=1
            while q:
                r,c = q.popleft()
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    rr,cc = r+dr,c+dc
                    if 0<=rr<H and 0<=cc<W and vis[rr,cc]==0 and out[rr,cc]==1:
                        vis[rr,cc]=1
                        q.append((rr,cc))
            out = vis
    return out

def compute_iou_per_class(pred, gt, num_classes=2):
    pred = torch.from_numpy(pred).long().view(-1)
    gt   = torch.from_numpy(gt).long().view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = pred==cls
        gt_inds   = gt==cls
        intersection = (pred_inds & gt_inds).sum().item()
        union = (pred_inds | gt_inds).sum().item()
        ious.append(float(intersection)/union if union>0 else float('nan'))
    return ious

def calculate_iou(pred, gt):
    ious = compute_iou_per_class(pred, gt, num_classes=2)
    bg_iou = ious[0]
    fg_iou = ious[1]
    miou = np.nanmean(ious)
    return bg_iou, fg_iou, miou

@torch.no_grad()
def validate_kmeans_and_save(loader, save_dir, k=16):
    os.makedirs(save_dir, exist_ok=True)
    bg_ious, fg_ious = [], []

    for images, scribbles, gts, names in loader:
        for img_t, scr_t, gt_t, name in zip(images, scribbles, gts, names):
            img_np = (img_t.permute(1,2,0).numpy()*255).astype(np.uint8)
            scr_np = scr_t.squeeze(0).numpy().astype(np.int16)
            gt_np  = gt_t.squeeze(0).numpy().astype(np.uint8)

            pred_mask = kmeans_segmentation(img_np, scr_np, k=k)
            # Post-processing to ensure scribbles are respected
            pred_mask = enforce_scribble_constraints(pred_mask, scr_np)

            # IoU
            bg_iou, fg_iou, miou = calculate_iou(pred_mask, gt_np)
            bg_ious.append(bg_iou)
            fg_ious.append(fg_iou)

            # Save
            save_path = os.path.join(save_dir, f"{name}.png")
            Image.fromarray((pred_mask*255).astype(np.uint8)).save(save_path)

    bg_mean = float(np.mean(bg_ious)) if bg_ious else 0.0
    fg_mean = float(np.mean(fg_ious)) if fg_ious else 0.0
    miou = np.nanmean([bg_mean, fg_mean])
    return bg_mean, fg_mean, miou

@torch.no_grad()
def generate_kmeans_predictions(loader, save_dir, k=16):
    os.makedirs(save_dir, exist_ok=True)
    for images, scribbles, _, names in loader:
        for img_t, scr_t, name in zip(images, scribbles, names):
            img_np = (img_t.permute(1,2,0).numpy()*255).astype(np.uint8)
            scr_np = scr_t.squeeze(0).numpy().astype(np.int16)
            pred_mask = kmeans_segmentation(img_np, scr_np, k=k)
            pred_mask = enforce_scribble_constraints(pred_mask, scr_np)
            save_path = os.path.join(save_dir, f"{name}.png")
            Image.fromarray((pred_mask*255).astype(np.uint8)).save(save_path)

if __name__ == "__main__":
    train_root = "dataset/train"
    test_root  = "dataset/test1"
    img_size = (256, 256)

    # --- K-Means Evaluation ---
    kmeans_save_dir = "dataset/predictions_kmeans"
    num_clusters = 16 # Hyperparameter to tune
    os.makedirs(kmeans_save_dir, exist_ok=True)

    # Note: Using a fresh instance of the dataloaders
    train_loader_km, val_loader_km = get_dataloaders(train_root, img_size, val_ratio=0.15, batch_size=1)

    bg_iou_train, fg_iou_train, miou_train = validate_kmeans_and_save(train_loader_km, save_dir=kmeans_save_dir, k=num_clusters)
    print(f"K-Means (k={num_clusters}) Train mIoU: {miou_train:.4f} (BG {bg_iou_train:.4f}, FG {fg_iou_train:.4f})")

    bg_iou_val, fg_iou_val, miou_val = validate_kmeans_and_save(val_loader_km, save_dir=kmeans_save_dir, k=num_clusters)
    print(f"K-Means (k={num_clusters}) Validation mIoU: {miou_val:.4f} (BG {bg_iou_val:.4f}, FG {fg_iou_val:.4f})")

    # --- Test1 predictions for K-Means ---
    test_ds_km = ScribbleSegmentationDataset(test_root, target_size=img_size, gt_dir=None, is_train=False)
    test_loader_km = DataLoader(test_ds_km, batch_size=1, shuffle=False)
    test_save_dir_km = os.path.join(kmeans_save_dir, "test1")
    generate_kmeans_predictions(test_loader_km, save_dir=test_save_dir_km, k=num_clusters)
    print(f"Test predictions for K-Means saved in {test_save_dir_km}")