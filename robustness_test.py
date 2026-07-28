import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

from data_loader import get_dataloaders
from train_unet import UNet, scribbles_to_input_channels, calculate_iou, enforce_scribble_constraints

from grab import grabcut_segmentation 

def add_gaussian_noise(image_np: np.ndarray, sigma=50) -> np.ndarray:
    """Adds Gaussian noise to an image."""
    row, col, ch = image_np.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image_np + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def reduce_brightness(image_np: np.ndarray, factor=0.3) -> np.ndarray:
    """Reduces the brightness of an image."""
    darkened = image_np.astype(np.float32) * factor
    return np.clip(darkened, 0, 255).astype(np.uint8)

@torch.no_grad()
def run_robustness_evaluation(unet_model, loader, device):
    unet_model.eval()
        scores = {
        "unet_original": [], "grabcut_original": [],
        "unet_noisy": [], "grabcut_noisy": [],
        "unet_dark": [], "grabcut_dark": []
    }

    print("Running robustness evaluation...")
    for images, scribbles, gts, _ in loader:
        for i in range(images.size(0)):
            image_t = images[i:i+1].to(device)
            scribble_t = scribbles[i:i+1].to(device)
            gt_np = gts[i].squeeze(0).cpu().numpy()
            
            image_np = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            scribble_np = scribbles[i].squeeze(0).cpu().numpy().astype(np.int16)
            image_noisy_np = add_gaussian_noise(image_np)
            image_dark_np = reduce_brightness(image_np)
            
            image_noisy_t = torch.from_numpy(image_noisy_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            image_dark_t = torch.from_numpy(image_dark_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            unet_pred_orig = (unet_model(torch.cat([image_t, scribbles_to_input_channels(scribble_t)], dim=1)) > 0.5)
            grabcut_pred_orig = grabcut_segmentation(image_np, scribble_np)

            unet_pred_noisy = (unet_model(torch.cat([image_noisy_t.to(device), scribbles_to_input_channels(scribble_t)], dim=1)) > 0.5)
            grabcut_pred_noisy = grabcut_segmentation(image_noisy_np, scribble_np)
            
            unet_pred_dark = (unet_model(torch.cat([image_dark_t.to(device), scribbles_to_input_channels(scribble_t)], dim=1)) > 0.5)
            grabcut_pred_dark = grabcut_segmentation(image_dark_np, scribble_np)


            def evaluate(pred, key):
                if isinstance(pred, torch.Tensor):
                    pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
                else: # It's already a numpy array from grabcut
                    pred_np = pred
                
                final_mask = enforce_scribble_constraints(pred_np, scribble_np)
                _, _, miou = calculate_iou(
                    torch.from_numpy(final_mask).unsqueeze(0).unsqueeze(0),
                    torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0)
                )
                scores[key].append(miou)
            
            evaluate(unet_pred_orig, "unet_original")
            evaluate(grabcut_pred_orig, "grabcut_original")
            evaluate(unet_pred_noisy, "unet_noisy")
            evaluate(grabcut_pred_noisy, "grabcut_noisy")
            evaluate(unet_pred_dark, "unet_dark")
            evaluate(grabcut_pred_dark, "grabcut_dark")

    print("\n--- Robustness Evaluation Complete ---")
    print(f"{'Model':<10} | {'mIoU (Original)':<18} | {'mIoU (Noisy)':<15} | {'mIoU (Dark)':<15}")
    print("-" * 65)
    
    unet_orig_avg = np.mean(scores['unet_original'])
    unet_noisy_avg = np.mean(scores['unet_noisy'])
    unet_dark_avg = np.mean(scores['unet_dark'])
    
    grab_orig_avg = np.mean(scores['grabcut_original'])
    grab_noisy_avg = np.mean(scores['grabcut_noisy'])
    grab_dark_avg = np.mean(scores['grabcut_dark'])

    print(f"{'U-Net':<10} | {unet_orig_avg:<18.4f} | {unet_noisy_avg:<15.4f} | {unet_dark_avg:<15.4f}")
    print(f"{'GrabCut':<10} | {grab_orig_avg:<18.4f} | {grab_noisy_avg:<15.4f} | {grab_dark_avg:<15.4f}")
    print("-" * 65)


if __name__ == "__main__":
    train_root = "dataset/train"
    img_size = (256, 256)
    val_ratio = 0.15
    batch_size = 1 # Process one image at a time for simplicity
    ckpt = "unet_scribble_5ch.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    _, val_loader = get_dataloaders(train_root, img_size, val_ratio, batch_size=batch_size, seed=42)

    unet_model = UNet(in_channels=5, out_channels=1, base_filters=32).to(device)
    unet_model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded U-Net model from {ckpt}")

    run_robustness_evaluation(unet_model, val_loader, device)
