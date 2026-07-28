# predict_with_unet.py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from util import load_dataset, store_predictions, visualize

# ====== 1) U-Net (same as in train_unet.py but 5-channel input) ======
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, base_filters=32):
        super().__init__()
        f=base_filters
        self.enc1=DoubleConv(in_channels,f); self.enc2=DoubleConv(f,f*2)
        self.enc3=DoubleConv(f*2,f*4); self.enc4=DoubleConv(f*4,f*8)
        self.pool=nn.MaxPool2d(2); self.bottleneck=DoubleConv(f*8,f*16)
        self.up4=nn.ConvTranspose2d(f*16,f*8,2,2); self.dec4=DoubleConv(f*16,f*8)
        self.up3=nn.ConvTranspose2d(f*8,f*4,2,2); self.dec3=DoubleConv(f*8,f*4)
        self.up2=nn.ConvTranspose2d(f*4,f*2,2,2); self.dec2=DoubleConv(f*4,f*2)
        self.up1=nn.ConvTranspose2d(f*2,f,2,2);   self.dec1=DoubleConv(f*2,f)
        self.final=nn.Conv2d(f,out_channels,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self.up4(b),e4],1))
        d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return torch.sigmoid(self.final(d1))

def _dilate_binary(mask: torch.Tensor, r: int = 2) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=2*r+1, stride=1, padding=r) if r>0 else mask

def scribbles_to_input_channels(scribbles: torch.Tensor, dilate_px: int = 2) -> torch.Tensor:
    fg = (scribbles==1).float(); bg=(scribbles==0).float()
    return torch.cat([_dilate_binary(fg,dilate_px), _dilate_binary(bg,dilate_px)], dim=1)

def enforce_scribble_constraints(pred_bin: np.ndarray, scribble_np: np.ndarray, keep_component=True) -> np.ndarray:
    from collections import deque
    out = pred_bin.copy()
    out[scribble_np==0]=0
    out[scribble_np==1]=1
    if keep_component:
        seeds = np.argwhere(scribble_np==1)
        if seeds.size>0:
            H,W = out.shape
            vis = np.zeros_like(out, dtype=np.uint8)
            q = deque([(int(r),int(c)) for r,c in seeds if out[int(r),int(c)]==1])
            for r,c in q: vis[r,c]=1
            while q:
                r,c=q.popleft()
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    rr,cc=r+dr,c+dc
                    if 0<=rr<H and 0<=cc<W and vis[rr,cc]==0 and out[rr,cc]==1:
                        vis[rr,cc]=1; q.append((rr,cc))
            out=vis
    return out

def segment_with_unet(image_np, scribble_np, model, device, img_size=(256,256)):
    """image_np: HxWx3 uint8, scribble_np: HxW in {0,1,255} (255=unlabeled). returns HxW in {0,1}."""
    H,W = scribble_np.shape[:2]
    # normalize scribble labels to {-1,0,1}
    scrib = scribble_np
    if scrib.ndim==3:  # if colored, take first channel or convert outside
        scrib = scrib[...,0]
    scrib = np.where(scrib==255, -1, scrib).astype(np.int16)

    # to tensors (resize like training)
    img_t = TF.to_tensor(TF.resize(Image.fromarray(image_np), img_size)).unsqueeze(0).to(device)
    scrib_r = Image.fromarray(scrib.astype(np.int16))
    scrib_t = torch.from_numpy(np.array(scrib_r.resize(img_size, resample=Image.NEAREST), dtype=np.int16)).unsqueeze(0).unsqueeze(0).float().to(device)
    scrib_ch = scribbles_to_input_channels(scrib_t, dilate_px=2)
    x = torch.cat([img_t, scrib_ch], dim=1)

    with torch.no_grad():
        pred = model(x)                                   # (1,1,h,w)
        pred = F.interpolate(pred, size=(H,W), mode='bilinear', align_corners=False)[0,0].cpu().numpy()
    binmask = (pred>0.5).astype(np.uint8)
    binmask = enforce_scribble_constraints(binmask, scrib, keep_component=True)
    return binmask

if __name__ == "__main__":
    # ---- paths / params ----
    ckpt_path = "unet_scribble_5ch.pth"   # <- the .pth you saved in training
    img_size  = (256,256)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(in_channels=5, out_channels=1, base_filters=32).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # # ---------- TRAIN SET (optional, for visualization) ----------
    images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
        "dataset/train", "images", "scribbles", "ground_truth"
    )
    # pred_train = np.stack(
    #     [segment_with_unet(img, scr, model, device, img_size) for img, scr in zip(images_train, scrib_train)],
    #     axis=0
    # )
    # store_predictions(pred_train, "dataset/train", "predictions_unet", fnames_train, palette)

    # ---------- TEST SET ----------
    images_test, scrib_test, fnames_test = load_dataset(
        "dataset/test1", "images", "scribbles"
    )
    pred_test = np.stack(
        [segment_with_unet(img, scr, model, device, img_size) for img, scr in zip(images_test, scrib_test)],
        axis=0
    )
    # NOTE: save into test1 (your snippet had "dataset/test" by mistake)
    store_predictions(pred_test, "dataset/test1", "predictions_unet", fnames_test, palette)
