# train_unet.py
import os
import glob
import random
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from collections import deque
from PIL import ImageFilter, ImageOps

# ---------------------------
# Utilities: IoU / metrics
# ---------------------------
def compute_iou_per_class(pred, gt, num_classes=2):
    pred = pred.view(-1)
    gt = gt.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        gt_inds = gt == cls
        intersection = (pred_inds & gt_inds).sum().item()
        union = (pred_inds | gt_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

def calculate_iou(pred, gt):
    ious = compute_iou_per_class(pred.squeeze(1).long(), gt.squeeze(1).long(), num_classes=2)
    bg_iou = ious[0]
    obj_iou = ious[1]
    miou = np.nanmean(ious)
    return bg_iou, obj_iou, miou
-
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


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            (in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f*2)
        self.enc3 = DoubleConv(f*2, f*4)
        self.enc4 = DoubleConv(f*4, f*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f*8, f*16)

        self.up4 = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f*16, f*8)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f*8, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f*2, f)

        self.final = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)

        out = self.final(d1)
        return torch.sigmoid(out)  # B,1,H,W


def _dilate_binary(mask: torch.Tensor, r: int = 2) -> torch.Tensor:
    """Max-pool dilation on a binary mask. mask: (B,1,H,W) in {0,1}"""
    if r <= 0: return mask
    return F.max_pool2d(mask, kernel_size=2*r+1, stride=1, padding=r)

def scribbles_to_input_channels(scribbles: torch.Tensor, dilate_px: int = 2) -> torch.Tensor:
    """
    Convert scribble tensor (B,1,H,W) with values {-1,0,1} to two binary channels:
      ch0 = FG_scribble (dilated), ch1 = BG_scribble (dilated)
    """
    fg = (scribbles == 1).float()
    bg = (scribbles == 0).float()
    fg = _dilate_binary(fg, r=dilate_px)
    bg = _dilate_binary(bg, r=dilate_px)
    return torch.cat([fg, bg], dim=1)  # (B,2,H,W)

def enforce_scribble_constraints(pred_bin: np.ndarray, scribble_np: np.ndarray, keep_component=True) -> np.ndarray:
    """
    pred_bin: (H,W) uint8 {0,1}
    scribble_np: (H,W) int {-1,0,1}

    1) force scribbled FG -> 1 and BG -> 0
    2) optionally keep only component(s) connected to any FG scribble seed
    """
    out = pred_bin.copy()
    out[scribble_np == 0] = 0
    out[scribble_np == 1] = 1

    if keep_component:
        seeds = np.argwhere(scribble_np == 1)
        if seeds.size > 0:
            H, W = out.shape
            vis = np.zeros_like(out, dtype=np.uint8)
            q = deque([(int(r), int(c)) for r, c in seeds if out[int(r), int(c)] == 1])
            for r, c in q:
                vis[r, c] = 1
            # 4-neighborhood BFS
            while q:
                r, c = q.popleft()
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < H and 0 <= cc < W and vis[rr, cc] == 0 and out[rr, cc] == 1:
                        vis[rr, cc] = 1
                        q.append((rr, cc))
            out = vis
    return out

# Losses
def dice_loss(pred, target, eps=1e-7):
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

def combined_loss(pred, gt, scribbles, ce_w=1.0, dice_w=1.0, scribble_w=0.5):
    """
    pred: B,1,H,W probabilities
    gt:   B,1,H,W (0/1) – full supervision
    scribbles: B,1,H,W with {-1,0,1}
    """
    ce = F.binary_cross_entropy(pred, gt)
    dl = dice_loss(pred, gt)

    # Scribble consistency: only labeled pixels; target = scribble label (0/1)
    labeled = (scribbles != -1)
    if labeled.any():
        scribble_target = torch.clamp(scribbles, 0, 1)  # {-1,0,1} -> {0,1} (unlabeled ignored)
        ce_scrib = F.binary_cross_entropy(pred[labeled], scribble_target[labeled])
    else:
        ce_scrib = torch.tensor(0.0, device=pred.device)

    return ce_w * ce + dice_w * dl + scribble_w * ce_scrib


# Train / Validate / Eval
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for images, scribbles, gts, _ in loader:
        images = images.to(device)
        scribbles = scribbles.to(device)
        gts = gts.to(device)

        scrib_in = scribbles_to_input_channels(scribbles, dilate_px=2)
        x = torch.cat([images, scrib_in], dim=1)  # B,5,H,W

        preds = model(x)  # B,1,H,W
        loss = combined_loss(preds, gts, scribbles, ce_w=1.0, dice_w=1.0, scribble_w=0.7)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu().item())
    return total / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ious = []
    for images, scribbles, gts, _ in loader:
        images = images.to(device)
        scribbles = scribbles.to(device)
        gts = gts.to(device)

        scrib_in = scribbles_to_input_channels(scribbles, dilate_px=2)
        x = torch.cat([images, scrib_in], dim=1)
        preds = model(x)
        preds_bin = (preds > 0.5).long()

        # optional hard enforcement to reflect test-time behavior
        out_np = preds_bin.squeeze(1).cpu().numpy().astype(np.uint8)
        scrib_np = scribbles.squeeze(1).cpu().numpy().astype(np.int16)
        for i in range(out_np.shape[0]):
            out_np[i] = enforce_scribble_constraints(out_np[i], scrib_np[i], keep_component=True)

        out_t = torch.from_numpy(out_np).unsqueeze(1).to(device)  # B,1,H,W
        bg_iou, obj_iou, miou = calculate_iou(out_t, gts)
        ious.append(miou)
    return float(np.mean(ious)) if ious else 0.0

@torch.no_grad()
def evaluate_miou(model, loader, device):
    """Full mIoU (BG and FG) averaging over loader, with scribble enforcement."""
    model.eval()
    total_bg, total_fg, n = 0.0, 0.0, 0
    for images, scribbles, gts, _ in loader:
        images = images.to(device)
        scribbles = scribbles.to(device)
        gts = gts.to(device)

        scrib_in = scribbles_to_input_channels(scribbles, dilate_px=2)
        x = torch.cat([images, scrib_in], dim=1)
        preds = model(x)
        preds_bin = (preds > 0.5).long()

        # enforce constraints + connectivity
        out_np = preds_bin.squeeze(1).cpu().numpy().astype(np.uint8)
        scrib_np = scribbles.squeeze(1).cpu().numpy().astype(np.int16)
        for i in range(out_np.shape[0]):
            out_np[i] = enforce_scribble_constraints(out_np[i], scrib_np[i], keep_component=True)

        out_t = torch.from_numpy(out_np).unsqueeze(1).to(device)  # B,1,H,W
        bg, fg, _ = calculate_iou(out_t, gts)
        total_bg += bg; total_fg += fg; n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    avg_bg, avg_fg = total_bg/n, total_fg/n
    return avg_bg, avg_fg, (avg_bg+avg_fg)/2.0


# Prediction / saving
@torch.no_grad()
def predict_and_save(model, root_folder, save_dir, device, img_size=(256,256)):
    """
    Run inference on a folder with images/ + scribbles/ (no gt required), save PNG masks.
    """
    os.makedirs(save_dir, exist_ok=True)
    ds = ScribbleSegmentationDataset(root_folder, target_size=img_size, gt_dir=None, is_train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    model.eval()
    for images, scribbles, _, names in loader:
        images = images.to(device)
        scribbles = scribbles.to(device)
        scrib_in = scribbles_to_input_channels(scribbles, dilate_px=2)
        x = torch.cat([images, scrib_in], dim=1)

        # simple TTA: horizontal flip
        preds1 = model(x)
        preds2 = model(torch.cat([TF.hflip(img.cpu()).unsqueeze(0) for img in x.cpu()], dim=0).to(device))
        preds2 = torch.cat([TF.hflip(p.squeeze(0)).unsqueeze(0) for p in preds2], dim=0)  # flip back
        preds = (preds1 + preds2) / 2.0

        preds_bin = (preds > 0.5).float()
        out_np = preds_bin.squeeze(1).cpu().numpy().astype(np.uint8)
        scrib_np = scribbles.squeeze(1).cpu().numpy().astype(np.int16)

        # enforce constraints + keep connected to FG seeds
        for i in range(out_np.shape[0]):
            out_np[i] = enforce_scribble_constraints(out_np[i], scrib_np[i], keep_component=True)
            Image.fromarray((out_np[i]*255).astype(np.uint8)).save(os.path.join(save_dir, f"{names[i]}.png"))


# ========= Influence visualization (scribble gradients) =========
import numpy as np
from PIL import Image
import os

def _norm01(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    m = a.max()
    return (a / m) if m > 0 else a

def _tensor_rgb_to_uint8(img3chw: torch.Tensor) -> np.ndarray:
    # (3,H,W) float[0,1] -> (H,W,3) uint8
    img = img3chw.detach().cpu().clamp(0,1).permute(1,2,0).numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)

def _scribble_tensor_to_uint8(scrib1hw: torch.Tensor) -> np.ndarray:
    # (1,H,W) in {-1,0,1} -> (H,W) uint8 in {255,0,1}
    s = scrib1hw.detach().cpu().numpy().astype(np.int16)[0]
    s = np.where(s == -1, 255, s).astype(np.uint8)
    return s

def overlay_scribbles(image_u8: np.ndarray, scrib_u8: np.ndarray,
                      fg_color=(0,255,0), bg_color=(255,0,0), alpha=0.7) -> np.ndarray:
    out = image_u8.copy()
    mfg = (scrib_u8 == 1); mbg = (scrib_u8 == 0)
    if mfg.any():
        out[mfg] = (alpha*np.array(fg_color) + (1-alpha)*out[mfg]).astype(np.uint8)
    if mbg.any():
        out[mbg] = (alpha*np.array(bg_color) + (1-alpha)*out[mbg]).astype(np.uint8)
    return out

def colorize_heatmap(heat01: np.ndarray, cmap_name="inferno") -> np.ndarray:
    import matplotlib.cm as cm
    colored = cm.get_cmap(cmap_name)(heat01)[..., :3]  # (H,W,3) float
    return (colored * 255.0 + 0.5).astype(np.uint8)

def overlay_heatmap(base_u8: np.ndarray, heat01: np.ndarray, alpha=0.55) -> np.ndarray:
    H, W, _ = base_u8.shape
    if heat01.shape[:2] != (H,W):
        heat01 = np.array(Image.fromarray((heat01*255).astype(np.uint8)).resize((W,H), Image.BILINEAR)) / 255.0
    hm_u8 = colorize_heatmap(heat01)  # (H,W,3)
    out = (alpha*hm_u8 + (1-alpha)*base_u8).astype(np.uint8)
    return out

def logits_from_prob(prob: torch.Tensor, eps=1e-6) -> torch.Tensor:
    p = prob.clamp(eps, 1-eps)
    return torch.log(p) - torch.log1p(-p)

def grad_scribble_influence_torch(model: nn.Module,
                                  img1_3hw: torch.Tensor,
                                  scrib1_1hw: torch.Tensor,
                                  dilate_px: int = 2) -> dict:
    """
    img1_3hw: (1,3,H,W) float[0,1]
    scrib1_1hw: (1,1,H,W) float in {-1,0,1}
    returns dict with 'fg','bg','combined' heatmaps in [0,1], each (H,W) np.ndarray
    """
    model.eval()
    H, W = img1_3hw.shape[-2:]

    S = scribbles_to_input_channels(scrib1_1hw, dilate_px=dilate_px)  # (1,2,H,W) in {0,1}
    S.requires_grad_(True)

    x = torch.cat([img1_3hw, S], dim=1)  # (1,5,H,W)
    prob = model(x)                      # (1,1,H,W)
    logit = logits_from_prob(prob)

    obj = logit.mean()
    model.zero_grad(set_to_none=True)
    if x.grad is not None: x.grad.zero_()
    obj.backward()

    g = S.grad.detach().abs()[0]         # (2,H,W)
    fg = _norm01(g[0].cpu().numpy())
    bg = _norm01(g[1].cpu().numpy())
    cb = _norm01(fg + bg)
    return {'fg': fg, 'bg': bg, 'combined': cb}

def save_influence_pngs(model: nn.Module,
                        val_loader: DataLoader,
                        device: torch.device,
                        out_dir: str,
                        epoch: int,
                        max_samples: int = 4,
                        dilate_px: int = 2):
    """
    Saves PNGs for up to max_samples from the first validation batch:
      - overlay of scribbles
      - FG, BG, and Combined influence overlays
    """
    os.makedirs(out_dir, exist_ok=True)
    # grab first batch
    try:
        images, scribbles, gts, names = next(iter(val_loader))
    except StopIteration:
        return
    n = min(max_samples, images.size(0))

    for i in range(n):
        img = images[i:i+1].to(device)         # (1,3,H,W)
        scr = scribbles[i:i+1].to(device)      # (1,1,H,W)
        name = names[i] if isinstance(names[i], str) else f"sample{i}"

        # base visuals
        base_u8 = _tensor_rgb_to_uint8(images[i])              # (H,W,3)
        scrib_u8 = _scribble_tensor_to_uint8(scribbles[i])     # (H,W) in {0,1,255}
        base_with_scrib = overlay_scribbles(base_u8, scrib_u8) # show strokes

        # compute influence
        infl = grad_scribble_influence_torch(model, img, scr, dilate_px=dilate_px)

        # overlays
        ov_fg = overlay_heatmap(base_with_scrib, infl['fg'])
        ov_bg = overlay_heatmap(base_with_scrib, infl['bg'])
        ov_cb = overlay_heatmap(base_with_scrib, infl['combined'])

        # save
        subdir = os.path.join(out_dir, f"epoch_{epoch:03d}")
        os.makedirs(subdir, exist_ok=True)
        Image.fromarray(base_with_scrib).save(os.path.join(subdir, f"{name}_00_scribbles.png"))
        Image.fromarray(ov_fg).save(os.path.join(subdir, f"{name}_01_infl_fg.png"))
        Image.fromarray(ov_bg).save(os.path.join(subdir, f"{name}_02_infl_bg.png"))
        Image.fromarray(ov_cb).save(os.path.join(subdir, f"{name}_03_infl_combined.png"))

def main():
    train_root = "dataset/train"      # has images/, scribbles/, ground_truth/
    test1_root = "dataset/test1"      # has images/, scribbles/ (no GT)
    img_size = (256, 256)
    epochs = 40
    batch_size = 8
    lr = 1e-3
    val_ratio = 0.15
    ckpt = "unet_scribble_5ch.pth"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Build full training dataset then split into train/val
    full_ds = ScribbleSegmentationDataset(train_root, target_size=img_size, is_train=True)
    n_val = max(1, int(len(full_ds)*val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(1234))

    # IMPORTANT: turn off augmentation for val subset
    val_ds.dataset.is_train = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"Train: {n_train} | Val: {n_val}")

    model = UNet(in_channels=5, out_channels=1, base_filters=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_bg, val_fg, val_miou = evaluate_miou(model, val_loader, device)
        print(f"Epoch {epoch:03d} | loss {train_loss:.4f} | mIoU {val_miou:.4f} (BG {val_bg:.4f}, FG {val_fg:.4f})")

        if val_miou > best_val:
            best_val = val_miou
            torch.save(model.state_dict(), ckpt)
            print(f"  -> Saved best to {ckpt}")

        if epoch % 5 == 0:  # change frequency as you like
                save_influence_pngs(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    out_dir="plots",     # top-level folder
                    epoch=epoch,
                    max_samples=4,       # save first 4 samples from the first val batch
                    dilate_px=2
                )
                print(f"Saved influence PNGs for epoch {epoch} under plots/epoch_{epoch:03d}/")
    print("Training finished. Best Val mIoU:", best_val)

    # Inference on Test1 (no GT): save predictions
    if os.path.exists(test1_root):
        save_dir = os.path.join(test1_root, "predictions_unet")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        predict_and_save(model, test1_root, save_dir, device, img_size=img_size)
        print("Saved Test1 predictions to:", save_dir)


if __name__ == "__main__":
    main()
