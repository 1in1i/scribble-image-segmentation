import os
import glob
import random
from collections import deque

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader

from data_loader import ScribbleSegmentationDataset, get_dataloaders

def grabcut_segmentation(image_np, scribble_np, iter_count=5):
    H, W, _ = image_np.shape
    mask = np.full((H, W), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[scribble_np == 0] = cv2.GC_BGD
    mask[scribble_np == 1] = cv2.GC_FGD
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (1,1,W-2,H-2)
    cv2.grabCut(image_np, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_MASK)
    output_mask = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
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
def validate_grabcut_and_save(loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    bg_ious, fg_ious = [], []

    for images, scribbles, gts, names in loader:
        for img_t, scr_t, gt_t, name in zip(images, scribbles, gts, names):
            img_np = (img_t.permute(1,2,0).numpy()*255).astype(np.uint8)
            scr_np = scr_t.squeeze(0).numpy().astype(np.int16)
            gt_np  = gt_t.squeeze(0).numpy().astype(np.uint8)

            pred_mask = grabcut_segmentation(img_np, scr_np, iter_count=5)
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
def generate_grabcut_predictions(loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for images, scribbles, _, names in loader:
        for img_t, scr_t, name in zip(images, scribbles, names):
            img_np = (img_t.permute(1,2,0).numpy()*255).astype(np.uint8)
            scr_np = scr_t.squeeze(0).numpy().astype(np.int16)
            pred_mask = grabcut_segmentation(img_np, scr_np, iter_count=5)
            pred_mask = enforce_scribble_constraints(pred_mask, scr_np)
            save_path = os.path.join(save_dir, f"{name}.png")
            Image.fromarray((pred_mask*255).astype(np.uint8)).save(save_path)

if __name__ == "__main__":
    train_root = "dataset/train"
    test_root  = "dataset/test1"
    img_size = (256, 256)
    save_dir = "dataset/predictions_grabcut2"
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader = get_dataloaders(train_root, img_size, val_ratio=0.15, batch_size=1)

    bg_iou_train, fg_iou_train, miou_train = validate_grabcut_and_save(train_loader, save_dir=save_dir)
    print(f"GrabCut Train mIoU: {miou_train:.4f} (BG {bg_iou_train:.4f}, FG {fg_iou_train:.4f})")

    bg_iou_val, fg_iou_val, miou_val = validate_grabcut_and_save(val_loader, save_dir=save_dir)
    print(f"GrabCut Validation mIoU: {miou_val:.4f} (BG {bg_iou_val:.4f}, FG {fg_iou_val:.4f})")

    # Test1 predictions
    test_ds = ScribbleSegmentationDataset(test_root, target_size=img_size, gt_dir=None, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    test_save_dir = os.path.join(save_dir, "test1")
    generate_grabcut_predictions(test_loader, save_dir=test_save_dir)
    print(f"Test predictions saved in {test_save_dir}")
