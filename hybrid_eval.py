import os
import numpy as np
from PIL import Image
import cv2  # Make sure to import cv2 for GrabCut

import torch
from torch.utils.data import DataLoader

# Import necessary components from your training script and data loader
from data_loader import get_dataloaders
from train_unet import UNet, scribbles_to_input_channels, evaluate_miou, calculate_iou, enforce_scribble_constraints

def hybrid_segmentation(image_np: np.ndarray, unet_mask_np: np.ndarray, iter_count: int = 3) -> np.ndarray:
    """
    Refines a U-Net segmentation mask using GrabCut by introducing uncertainty at the boundary.
    """
    # Create a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    
    # 1. Erode the U-Net mask to find the "definite foreground" (core of the object)
    sure_fg = cv2.erode(unet_mask_np, kernel, iterations=2)
    
    # 2. Dilate the U-Net mask to find the "probable foreground" (boundary region)
    probable_fg = cv2.dilate(unet_mask_np, kernel, iterations=3)
    
    # 3. Initialize the GrabCut mask
    mask = np.full(image_np.shape[:2], cv2.GC_BGD, dtype=np.uint8) # Everything is initially background
    
    # Pixels in the probable_fg region but not in the sure_fg core are uncertain
    mask[probable_fg == 1] = cv2.GC_PR_FGD # Probable Foreground
    
    # Pixels in the sure_fg core are definitely foreground
    mask[sure_fg == 1] = cv2.GC_FGD      # Definite Foreground

    # 4. Run GrabCut to refine the uncertain (probable) regions
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(image_np, mask, None, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_MASK)

    # 5. Create the final binary mask
    output_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    
    return output_mask

@torch.no_grad()
def evaluate_hybrid_model(unet_model, loader, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    unet_model.eval()
    
    print("Running evaluation for Hybrid (U-Net + GrabCut) model...")

    # --- Step 1: Evaluate the U-Net alone ---
    all_unet_preds = []
    all_gts = []
    for images, scribbles, gts, _ in loader:
        images_t = images.to(device); scribbles_t = scribbles.to(device)
        scrib_in = scribbles_to_input_channels(scribbles_t, dilate_px=2)
        x = torch.cat([images_t, scrib_in], dim=1)
        unet_preds_bin = (unet_model(x) > 0.5).long()

        unet_np_raw = unet_preds_bin.squeeze(1).cpu().numpy().astype(np.uint8)
        scribble_np = scribbles.squeeze(1).cpu().numpy().astype(np.int16)
        for i in range(unet_np_raw.shape[0]):
            final_mask = enforce_scribble_constraints(unet_np_raw[i], scribble_np[i])
            all_unet_preds.append(torch.from_numpy(final_mask))
            all_gts.append(gts[i])
    
    from torch.utils.data import TensorDataset
    unet_results_ds = TensorDataset(torch.stack(all_gts), torch.stack(all_unet_preds))
    unet_results_loader = DataLoader(unet_results_ds, batch_size=loader.batch_size)
    
    # Capture all three return values for the U-Net
    avg_unet_bg_iou, avg_unet_fg_iou, avg_unet_miou = evaluate_miou_from_tensors(unet_results_loader, device)

    # --- Step 2: Evaluate the Hybrid Model ---
    all_hybrid_preds = []
    # Re-use the ground truth tensors from the first loop
    # all_gts = [] 
    for images, scribbles, gts, names in loader:
        images_t = images.to(device); scribbles_t = scribbles.to(device)
        scrib_in = scribbles_to_input_channels(scribbles_t, dilate_px=2)
        x = torch.cat([images_t, scrib_in], dim=1)
        unet_preds_bin = (unet_model(x) > 0.5).long()

        for i in range(images.size(0)):
            image_np = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            unet_mask_np = unet_preds_bin[i].squeeze(0).cpu().numpy().astype(np.uint8)
            scribble_np = scribbles[i].squeeze(0).cpu().numpy().astype(np.int16)
            
            hybrid_mask_np = hybrid_segmentation(image_np, unet_mask_np, iter_count=3)
            hybrid_mask_final = enforce_scribble_constraints(hybrid_mask_np, scribble_np)
            all_hybrid_preds.append(torch.from_numpy(hybrid_mask_final))

            save_path = os.path.join(save_dir, f"{names[i]}.png")
            Image.fromarray((hybrid_mask_final * 255).astype(np.uint8)).save(save_path)

    hybrid_results_ds = TensorDataset(torch.stack(all_gts), torch.stack(all_hybrid_preds))
    hybrid_results_loader = DataLoader(hybrid_results_ds, batch_size=loader.batch_size)
    
    # Capture all three return values for the Hybrid model
    avg_hybrid_bg_iou, avg_hybrid_fg_iou, avg_hybrid_miou = evaluate_miou_from_tensors(hybrid_results_loader, device)

    print("\n--- Evaluation Complete (Consistent Method) ---")
    print(f"U-Net Final:      mIoU {avg_unet_miou:.4f} (BG {avg_unet_bg_iou:.4f}, FG {avg_unet_fg_iou:.4f})")
    print(f"Hybrid Model:     mIoU {avg_hybrid_miou:.4f} (BG {avg_hybrid_bg_iou:.4f}, FG {avg_hybrid_fg_iou:.4f})")
    print("---------------------------------------------")
    if avg_hybrid_miou > avg_unet_miou:
        print("Hybrid model shows an improvement!")
    else:
        print("ℹHybrid model did not improve over U-Net alone.")

@torch.no_grad()
def evaluate_miou_from_tensors(loader, device):
    total_bg, total_fg, n = 0.0, 0.0, 0
    for gts, preds in loader:
        gts = gts.to(device)
        preds = preds.unsqueeze(1).to(device) # Add channel dim
        bg, fg, _ = calculate_iou(preds, gts)
        if not np.isnan(bg) and not np.isnan(fg):
            total_bg += bg
            total_fg += fg
            n += gts.size(0) # Use actual number of items
    if n == 0:
        return 0.0, 0.0, 0.0
    # This is batch-weighted average, which is more stable
    avg_bg, avg_fg = total_bg/len(loader), total_fg/len(loader) 
    return avg_bg, avg_fg, (avg_bg + avg_fg) / 2.0


if __name__ == "__main__":
    train_root = "dataset/train"
    img_size = (256, 256)
    val_ratio = 0.15
    batch_size = 8  # Can be smaller if you run out of memory
    ckpt = "unet_scribble_5ch.pth"  # Your best U-Net checkpoint
    save_dir = "dataset/predictions_hybrid"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # We only need the validation loader for this evaluation
    _, val_loader = get_dataloaders(train_root, img_size, val_ratio, batch_size)

    model = UNet(in_channels=5, out_channels=1, base_filters=32).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded U-Net model from {ckpt}")

    evaluate_hybrid_model(model, val_loader, device, save_dir)