import os
import glob
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF

# Make sure to copy the following classes and functions into this file:
# ScribbleSegmentationDataset, _to_scribble_tensor, _to_gt_tensor, _paired_random_geo_aug, _heavy_color_aug
# The functions for paired augmentation are already part of ScribbleSegmentationDataset
# The imports should also be copied.

class ScribbleSegmentationDataset(Dataset):
    """
    Folder structure:
      root/images/*.png|jpg
      root/scribbles/*.png|jpg   (0 = BG-scribble, 1 = FG-scribble, 255 = unmarked)
      root/ground_truth/*.png    (0/1)  [optional for test]

    Returns:
      img:    FloatTensor (3,H,W) in [0,1]
      scrib:  FloatTensor (1,H,W) with values {-1,0,1}
      gt:     FloatTensor (1,H,W) in {0,1} (zeros if gt_dir=None)
      name:   str (basename without extension)
    """

    def __init__(self,
                 root_folder: str,
                 images_dir="images",
                 scribbles_dir="scribbles",
                 gt_dir="ground_truth",
                 target_size=(256, 256),
                 is_train: bool = True):
        self.root = root_folder
        self.images_dir = os.path.join(root_folder, images_dir)
        self.scribbles_dir = os.path.join(root_folder, scribbles_dir)
        self.gt_dir = os.path.join(root_folder, gt_dir) if gt_dir is not None else None

        self.image_paths = sorted([p for p in glob.glob(os.path.join(self.images_dir, "*")) if not os.path.basename(p).startswith('.')])
        self.scribble_paths = sorted([p for p in glob.glob(os.path.join(self.scribbles_dir, "*")) if not os.path.basename(p).startswith('.')])

        if self.gt_dir:
            self.gt_paths = sorted([p for p in glob.glob(os.path.join(self.gt_dir, "*")) if not os.path.basename(p).startswith('.')])
            assert len(self.image_paths) == len(self.gt_paths) == len(self.scribble_paths), "Mismatched counts."
        else:
            self.gt_paths = None
            assert len(self.image_paths) == len(self.scribble_paths), "Mismatched counts."

        self.target_size = target_size
        self.is_train = is_train

        # Color jitter only for the RGB image
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

    def __len__(self):
        return len(self.image_paths)

    def _load_triplet(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        scrib = Image.open(self.scribble_paths[idx]).convert("L")
        if self.gt_paths is not None:
            gt = Image.open(self.gt_paths[idx]).convert("L")
        else:
            gt = None
        return img, scrib, gt

    def _to_scribble_tensor(self, scrib_img: Image.Image, size: Tuple[int,int]):
        scrib = scrib_img.resize(size, resample=Image.NEAREST)
        scrib_np = np.array(scrib, dtype=np.int16)  # 0,1,255
        scrib_np = np.where(scrib_np == 255, -1, scrib_np)  # -1 unlabeled
        return torch.from_numpy(scrib_np).unsqueeze(0).float()  # (1,H,W)

    def _to_gt_tensor(self, gt_img: Image.Image, size: Tuple[int,int]):
        if gt_img is None:
            return None
        gt = gt_img.resize(size, resample=Image.NEAREST)
        gt_np = (np.array(gt, dtype=np.uint8) > 0).astype(np.float32)
        return torch.from_numpy(gt_np).unsqueeze(0).float()  # (1,H,W)

    def _paired_random_geo_aug(self, img, scrib, gt):
        """Apply the same geometric aug to image/scrib/gt."""
        # random horizontal flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            scrib = TF.hflip(scrib)
            if gt is not None:
                gt = TF.hflip(gt)
       # random rotation ±30°
        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        scrib = TF.rotate(scrib, angle, interpolation=TF.InterpolationMode.NEAREST)
        if gt is not None:
            gt = TF.rotate(gt, angle, interpolation=TF.InterpolationMode.NEAREST)

        # random scaling 0.7–1.3
        scale = random.uniform(0.7, 1.3)
        w, h = img.size
        new_w, new_h = int(w*scale), int(h*scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        scrib = scrib.resize((new_w, new_h), Image.NEAREST)
        if gt is not None:
            gt = gt.resize((new_w, new_h), Image.NEAREST)

        # random translation ±10%
        max_dx = int(0.1*new_w)
        max_dy = int(0.1*new_h)
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        img = TF.affine(img, angle=0, translate=(dx, dy), scale=1.0, shear=0, interpolation=TF.InterpolationMode.BILINEAR)
        scrib = TF.affine(scrib, angle=0, translate=(dx, dy), scale=1.0, shear=0, interpolation=TF.InterpolationMode.NEAREST)
        if gt is not None:
            gt = TF.affine(gt, angle=0, translate=(dx, dy), scale=1.0, shear=0, interpolation=TF.InterpolationMode.NEAREST)

        return img, scrib, gt
    
    def _heavy_color_aug(self, img):
        # ColorJitter
        color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05
        )
        img = color_jitter(img)

        # random gray 10%
        if random.random() < 0.1:
            img = TF.to_grayscale(img, num_output_channels=3)

        # Gaussian blur 50% 
        if random.random() < 0.5:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius))
        
        # random sharp 30% 
        if random.random() < 0.3:
            factor = random.uniform(1.0, 2.0)  
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(factor*150), threshold=3))
        
        return img

    def __getitem__(self, idx):
        img, scrib, gt = self._load_triplet(idx)
        img_name = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]

        if self.is_train:
            # heavy geometric aug
            img, scrib, gt = self._paired_random_geo_aug(img, scrib, gt)
            # heavy color/noise aug
            img = self._heavy_color_aug(img)

        # resize to target
        img = TF.resize(img, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        scrib_tensor = self._to_scribble_tensor(scrib, self.target_size)
        gt_tensor = self._to_gt_tensor(gt, self.target_size)
        if gt_tensor is None:
            gt_tensor = torch.zeros_like(scrib_tensor)

        img_t = TF.to_tensor(img)  # (3,H,W) in [0,1]
        return img_t, scrib_tensor, gt_tensor, img_name


def get_dataloaders(train_root: str, img_size: Tuple[int, int], val_ratio: float, batch_size: int, seed: int = 1234):
    """
    Creates and returns data loaders for training and validation.
    
    Args:
        train_root (str): Path to the training dataset root folder.
        img_size (tuple): Target image size (height, width).
        val_ratio (float): The ratio of the dataset to use for validation.
        batch_size (int): The batch size for the data loaders.
        seed (int): The random seed for the data split.

    Returns:
        A tuple containing: (train_loader, val_loader)
    """
    full_ds = ScribbleSegmentationDataset(train_root, target_size=img_size, is_train=True)
    n_val = max(1, int(len(full_ds) * val_ratio))
    n_train = len(full_ds) - n_val
    
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    # IMPORTANT: turn off augmentation for val subset
    val_ds.dataset.is_train = False
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"Train: {n_train} | Val: {n_val}")
    
    return train_loader, val_loader
