import os
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Import all necessary functions from your existing scripts
from train_unet import UNet, scribbles_to_input_channels, enforce_scribble_constraints
from grab import grabcut_segmentation
# from kmeans import kmeans_segmentation # Assuming you have a kmeans.py
from hybrid_eval import hybrid_segmentation # Assuming you have hybrid_eval.py

# ---------------------------
# Visualization Helpers
# ---------------------------
def overlay_scribbles(image_np, scribble_np):
    """Overlays scribbles on an image for visualization."""
    vis_img = image_np.copy()
    fg_color = [0, 255, 0]  # Green for foreground
    bg_color = [255, 0, 0]  # Red for background
    vis_img[scribble_np == 1] = fg_color
    vis_img[scribble_np == 0] = bg_color
    return vis_img

def create_red_on_black_mask(binary_mask_np):
    """Creates a red foreground on a black background visualization."""
    h, w = binary_mask_np.shape
    vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
    vis_mask[binary_mask_np == 1] = [128, 0, 0] # Dark Red color
    return vis_mask
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

    # --- FIX ADDED HERE ---
    # Ensure the scribble mask is a single-channel (2D) array.
    # The error indicates it's likely a 3-channel (3D) array.
    if scribble_np.ndim == 3:
        scribble_np = scribble_np[:, :, 0]
    # --------------------

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
# ---------------------------
# Main Figure Generation Logic
# ---------------------------
def generate_comparison_figure(config):
    # --- 1. Load Data ---
    img_path = os.path.join(config["data_root"], "images", config["image_name"] + ".jpg")
    scribble_path = os.path.join(config["data_root"], "scribbles", config["image_name"] + ".png")
    
    if not os.path.exists(img_path) or not os.path.exists(scribble_path):
        print(f"Error: Could not find image or scribble for {config['image_name']}")
        return

    original_img = Image.open(img_path).convert("RGB")
    original_scribble = Image.open(scribble_path)

    # --- 2. Preprocess Data ---
    img_resized = original_img.resize(config["img_size"])
    scribble_resized = original_scribble.resize(config["img_size"], Image.NEAREST)

    img_np = np.array(img_resized)
    scribble_np_raw = np.array(scribble_resized)
    
    # Ensure the raw scribble is single-channel before processing
    if scribble_np_raw.ndim == 3:
        scribble_np_raw = scribble_np_raw[:, :, 0]

    # Convert scribble to the format used by your models {-1, 0, 1}
    scribble_np = np.full(scribble_np_raw.shape, -1, dtype=np.int16)
    scribble_np[scribble_np_raw == 0] = 0 # Background
    scribble_np[scribble_np_raw == 1] = 1 # Foreground

    # Prepare tensor versions for U-Net
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    scribble_t = torch.from_numpy(scribble_np).float().unsqueeze(0).unsqueeze(0)

    # --- 3. Load U-Net Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = UNet(in_channels=5, out_channels=1, base_filters=32).to(device)
    unet_model.load_state_dict(torch.load(config["unet_ckpt"], map_location=device))
    unet_model.eval()

    # --- 4. Generate All Predictions ---
    print("Generating predictions...")
    
    # K-Means
    kmeans_pred_raw = kmeans_segmentation(img_np, scribble_np, k=config["kmeans_k"])
    kmeans_pred = enforce_scribble_constraints(kmeans_pred_raw, scribble_np)

    # GrabCut
    grabcut_pred_raw = grabcut_segmentation(img_np, scribble_np, iter_count=config["grabcut_iters"])
    grabcut_pred = enforce_scribble_constraints(grabcut_pred_raw, scribble_np)

    # U-Net
    with torch.no_grad():
        scrib_in = scribbles_to_input_channels(scribble_t.to(device), dilate_px=2)
        x = torch.cat([img_t.to(device), scrib_in], dim=1)
        unet_pred_raw = (unet_model(x) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    unet_pred = enforce_scribble_constraints(unet_pred_raw, scribble_np)

    # Hybrid Model
    hybrid_pred_raw = hybrid_segmentation(img_np, unet_pred_raw, iter_count=3)
    hybrid_pred = enforce_scribble_constraints(hybrid_pred_raw, scribble_np)
    
    print("Predictions generated.")

    # --- 5. Create and Save Visualizations ---
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # Save red-on-black versions for all models
    models_to_save = {
        "kmeans": kmeans_pred,
        "grabcut": grabcut_pred,
        "unet": unet_pred,
        "hybrid": hybrid_pred
    }

    for model_name, prediction_mask in models_to_save.items():
        red_black_vis = create_red_on_black_mask(prediction_mask)
        
        # Resize the red-on-black image to 500x375
        red_black_img = Image.fromarray(red_black_vis).resize((500, 375), Image.NEAREST)
        
        save_path = os.path.join(config["save_dir"], f"{config['image_name']}_{model_name}_red_black.png")
        red_black_img.save(save_path)
        print(f"Saved {model_name} red-on-black visualization to {save_path}")

    # Create the comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.suptitle(f"Model Predictions for {config['image_name']}.jpg", fontsize=16)

    # Row 1
    axes[0, 0].imshow(overlay_scribbles(img_np, scribble_np))
    axes[0, 0].set_title("Original Image + Scribbles")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(kmeans_pred, cmap='gray')
    axes[0, 1].set_title("K-Means Prediction")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(grabcut_pred, cmap='gray')
    axes[0, 2].set_title("GrabCut Prediction")
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(unet_pred, cmap='gray')
    axes[1, 0].set_title("U-Net Prediction")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(hybrid_pred, cmap='gray')
    axes[1, 1].set_title("Hybrid (U-Net + GrabCut)")
    axes[1, 1].axis('off')
    
    # Show the red-on-black for the hybrid in the plot (or pick one)
    axes[1, 2].imshow(create_red_on_black_mask(hybrid_pred))
    axes[1, 2].set_title("Hybrid (Red/Black)")
    axes[1, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_comp = os.path.join(config["save_dir"], f"{config['image_name']}_comparison.png")
    plt.savefig(save_path_comp, dpi=200)
    print(f"Saved comparison figure to {save_path_comp}")
    plt.show()


if __name__ == "__main__":
    # === CONFIGURATION ===
    config = {
        "data_root": "dataset/train",      # Path to your training data
        "image_name": "2008_005105",        # The specific image to process
        "img_size": (256, 256),
        "unet_ckpt": "unet_scribble_5ch.pth", # Path to your best U-Net checkpoint
        "save_dir": "figures",             # Directory to save the output images
        "kmeans_k": 32,                    # Best k for K-Means
        "grabcut_iters": 5,                # Best iter_count for GrabCut
    }
    # =====================
    
    generate_comparison_figure(config)