# %% [markdown]
# # My Own Fast Frame Prediction Model
# 
# A lightweight residual temporal U-Net for Inter4K GOP-8 Y-channel frame prediction. It keeps the same input/output format as `model2.ipynb`: `(B, 7, 1, H, W) -> (B, 1, H, W)`.
# 
# Design goal: faster training than the ConvNeXt/Transformer reference while still targeting about 30 dB validation PSNR after training.

# %%
import os, math, random, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # Windows local workaround for duplicate OpenMP runtimes
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

ON_KAGGLE = os.path.exists('/kaggle/working')

# for item in Path("/kaggle/working").iterdir():
#     if item.is_file():
#         item.unlink()
#     elif item.is_dir():
#         shutil.rmtree(item)

if ON_KAGGLE:
    DATASET_ROOT = '/kaggle/input/datasets/tonmoyk983/sevtone-4-qp-gop8/sevtone_4_QP_GOP8/Inter4K/RAW'
    CKPT_DIR = '/kaggle/working'
else:
    DATASET_ROOT = r'D:\\Dataset\\Inter4K\\60fps\\UHD\\Segments\\sevtone_4_QP_GOP8\\Inter4K\\RAW'
    CKPT_DIR = r'D:\\Dataset\\Inter4K\\60fps\\UHD\\Segments\\sevtone_4_QP_GOP8'

CKPT_DIR = Path(CKPT_DIR)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CKPT_DIR / 'myown_fast_best_model.pth'
LAST_CKPT_PATH = CKPT_DIR / 'myown_fast_last_checkpoint.pth'

# Dataset / training config
T = 7
IMG_SIZE = 512
SUBSET_SIZE = 20000
VAL_FRACTION = 0.1
TRAIN_CROP_SIZE = 512   # set to 512 for full-image training; 256 is much faster

# Model / optimizer config
BASE_CHANNELS = 48       # increase to 64 for more PSNR, decrease to 32 for more speed
BATCH_SIZE = 4           # try 12/16 if your GPU has room
EPOCHS = 40
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
USE_AMP = True
USE_TQDM = False
BATCH_SUMMARY_EVERY = 100 
NUM_WORKERS = 2
RESUME_CKPT_PATH = "/kaggle/input/models/vaselinek983/check3/pytorch/3/1/myown_fast_last_checkpoint.pth"  # or str(LAST_CKPT_PATH)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

print(f'Running on: {"Kaggle" if ON_KAGGLE else "Local"}')
print(f'DATASET_ROOT : {DATASET_ROOT}')
print(f'Device       : {DEVICE}')
print(f'Best model   : {BEST_MODEL_PATH}')

# %% [markdown]
# ## Dataset
# 
# The loader follows `model2.ipynb`: input files are `sample_input_<id>.npy` with shape `(7, 512, 512)`, and target files are `sample_output_<id>.npy` with shape `(1, 512, 512)` or `(512, 512)`.

# %%
class Inter4KDataset(Dataset):
    def __init__(self, root, sample_ids, crop_size=None, random_crop=True):
        self.input_dir = Path(root) / 'Input'
        self.output_dir = Path(root) / 'Output'
        self.ids = list(sample_ids)
        self.crop_size = crop_size
        self.random_crop = random_crop

    def __len__(self):
        return len(self.ids)

    def _crop_pair(self, x, y):
        if self.crop_size is None:
            return x, y
        _, h, w = x.shape
        cs = min(self.crop_size, h, w)
        if self.random_crop:
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
        else:
            top = (h - cs) // 2
            left = (w - cs) // 2
        return x[:, top:top + cs, left:left + cs], y[:, top:top + cs, left:left + cs]

    def __getitem__(self, idx):
        sid = self.ids[idx]
        x = np.load(self.input_dir / f'sample_input_{sid}.npy')
        y = np.load(self.output_dir / f'sample_output_{sid}.npy')
        if y.ndim == 2:
            y = y[np.newaxis]

        x, y = self._crop_pair(x, y)
        x = np.ascontiguousarray(x.astype(np.float32) / 255.0)
        y = np.ascontiguousarray(y.astype(np.float32) / 255.0)

        x = torch.from_numpy(x).unsqueeze(1)  # (7, 1, H, W)
        y = torch.from_numpy(y)               # (1, H, W)
        return x, y


all_ids = list(range(1, SUBSET_SIZE + 1))
split_idx = int(len(all_ids) * (1 - VAL_FRACTION))
train_ids = all_ids[:split_idx]
val_ids = all_ids[split_idx:]

train_ds = Inter4KDataset(DATASET_ROOT, train_ids, crop_size=TRAIN_CROP_SIZE, random_crop=True)
val_ds = Inter4KDataset(DATASET_ROOT, val_ids, crop_size=None, random_crop=False)

loader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == 'cuda'),
    persistent_workers=(NUM_WORKERS > 0),
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_ds, batch_size=max(1, BATCH_SIZE // 2), shuffle=False, **loader_kwargs)

print(f'Train: {len(train_ds)} samples | Val: {len(val_ds)} samples')
print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')
x0, y0 = train_ds[0]
print(f'Sample x shape: {tuple(x0.shape)} | y shape: {tuple(y0.shape)}')

# %% [markdown]
# ## Fast Model
# 
# This model uses temporal channel stacking, first-order frame differences, a small two-level U-Net, depthwise-separable residual blocks, and residual prediction from the last input frame.

# %%
class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DSResBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )

    def forward(self, x):
        return x + self.net(x)


class FastTemporalUNet(nn.Module):
    def __init__(self, T=7, base_ch=48):
        super().__init__()
        self.T = T
        in_ch = T + (T - 1)  # raw frames plus temporal differences

        self.stem = nn.Sequential(
            ConvGNAct(in_ch, base_ch),
            DSResBlock(base_ch),
        )
        self.down1 = nn.Sequential(
            ConvGNAct(base_ch, base_ch * 2, stride=2),
            DSResBlock(base_ch * 2),
            DSResBlock(base_ch * 2),
        )
        self.down2 = nn.Sequential(
            ConvGNAct(base_ch * 2, base_ch * 3, stride=2),
            DSResBlock(base_ch * 3),
            DSResBlock(base_ch * 3),
            DSResBlock(base_ch * 3),
        )
        self.up1 = nn.Sequential(
            ConvGNAct(base_ch * 3 + base_ch * 2, base_ch * 2),
            DSResBlock(base_ch * 2),
        )
        self.up2 = nn.Sequential(
            ConvGNAct(base_ch * 2 + base_ch, base_ch),
            DSResBlock(base_ch),
        )
        self.head = nn.Conv2d(base_ch, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, T, 1, H, W)
        B, T_in, C, H, W = x.shape
        assert T_in == self.T and C == 1, f'Expected (B, {self.T}, 1, H, W), got {tuple(x.shape)}'
        frames = x.squeeze(2)                 # (B, T, H, W)
        diffs = frames[:, 1:] - frames[:, :-1] # (B, T-1, H, W)
        z = torch.cat([frames, diffs], dim=1)
        last = x[:, -1]

        s0 = self.stem(z)
        s1 = self.down1(s0)
        s2 = self.down2(s1)

        u1 = F.interpolate(s2, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, s1], dim=1))
        u2 = F.interpolate(u1, size=s0.shape[-2:], mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, s0], dim=1))

        delta = self.head(u2)
        return torch.clamp(last + delta, 0.0, 1.0)


model = FastTemporalUNet(T=T, base_ch=BASE_CHANNELS)
model = model.to(DEVICE)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parameters: {total:,} total | {trainable:,} trainable')

# %%
model.eval()
with torch.no_grad():
    dummy = torch.randn(2, T, 1, TRAIN_CROP_SIZE, TRAIN_CROP_SIZE, device=DEVICE)
    out = model(dummy)
print(f'Input shape : {tuple(dummy.shape)}')
print(f'Output shape: {tuple(out.shape)}')
assert out.shape == (2, 1, TRAIN_CROP_SIZE, TRAIN_CROP_SIZE)
print('Shape verification passed')


# %% [markdown]
# ## Prediction Time Benchmark on Actual Videos
#
# Measure inference time for predicting frames 8, 16, 24, and 32 on the video classes.

import time
import re
import csv

print('\n--- Prediction Time Benchmark ---')

# Load weights if available
EVAL_MODEL_PATH = CKPT_DIR / 'myown_fast_best_model_final.pth'
if not EVAL_MODEL_PATH.exists():
    EVAL_MODEL_PATH = BEST_MODEL_PATH

if EVAL_MODEL_PATH.exists():
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_model.load_state_dict(torch.load(EVAL_MODEL_PATH, map_location=DEVICE))
    print(f'Loaded checkpoint: {EVAL_MODEL_PATH}')
else:
    print('No checkpoint found. Running with untrained weights.')

model.eval()

# You can change this to point to the root containing all class folders.
TESTING_CLASSD2_ROOT = Path('./segments/All_Sequence')
INFERENCE_SIZE = 512
GOP_SIZE = 8

def read_yuv420_y_frames(yuv_path, width, height):
    frame_size = width * height * 3 // 2
    raw = np.fromfile(yuv_path, dtype=np.uint8)
    if raw.size % frame_size != 0:
        raise ValueError(f'File size mismatch for {yuv_path} (expected multiples of {frame_size} bytes per frame)')
    num_frames = raw.size // frame_size
    y_size = width * height
    frames = raw.reshape(num_frames, frame_size)
    y_frames = frames[:, :y_size].reshape(num_frames, height, width)
    return y_frames

def find_resolution_from_path(path):
    for part in [path.name, *path.parts]:
        match = re.search(r'(\d+)x(\d+)', part)
        if match:
            return int(match.group(1)), int(match.group(2))
    raise ValueError(f'Could not parse resolution from path {path}')

def resize_frame_stack(frames, size=INFERENCE_SIZE):
    tensor = torch.from_numpy(frames).unsqueeze(1).float()
    tensor = F.interpolate(tensor, size=(size, size), mode='bilinear', align_corners=False)
    return tensor.squeeze(1).cpu().numpy()

def load_frame_stack(yuv_path):
    width, height = find_resolution_from_path(yuv_path)
    frames = read_yuv420_y_frames(yuv_path, width, height).astype(np.float32) / 255.0
    return resize_frame_stack(frames)


target_frames = [8, 16, 24, 32]
total_time = 0.0

print(f'Testing root: {TESTING_CLASSD2_ROOT}')

# Warm up
dummy = torch.randn(1, 7, 1, 512, 512, device=DEVICE)
with torch.no_grad():
    _ = model(dummy)
    if DEVICE.type == 'cuda': torch.cuda.synchronize(DEVICE)

out_csv = CKPT_DIR / 'prediction_times.csv'
print(f'Saving prediction times to: {out_csv}')
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['video_id', 'qp', 'yuv_file', 'frame_index', 'pred_time_sec'])

with torch.no_grad():
    for recon_yuv_path in sorted(TESTING_CLASSD2_ROOT.rglob('*.yuv')):
        # Handle both flat structure and QP_* subfolder structure
        video_id = recon_yuv_path.stem
        qp = recon_yuv_path.parent.name if recon_yuv_path.parent.name.startswith('QP_') else 'N/A'
        class_name = recon_yuv_path.parent.parent.name if qp != 'N/A' else recon_yuv_path.parent.name
        
        print(f"\nProcessing: {class_name} - {video_id} - {qp} ({recon_yuv_path.name})")
        
        try:
            recon_frames = load_frame_stack(recon_yuv_path)
        except Exception as e:
            print(f"  Failed to load {recon_yuv_path.name}: {e}")
            continue
        
        if recon_frames.shape[0] < max(target_frames):
            print(f"  Not enough frames (found {recon_frames.shape[0]}, need {max(target_frames)})")
            continue
        
        for frame_index in target_frames:
            start = frame_index - GOP_SIZE
            end = frame_index - 1
            
            x_np = recon_frames[start:end]
            x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(2).to(DEVICE)
            
            if DEVICE.type == 'cuda': torch.cuda.synchronize(DEVICE)
            start_t = time.perf_counter()
            
            pred = base_model(x).float()
            
            if DEVICE.type == 'cuda': torch.cuda.synchronize(DEVICE)
            pred_time = time.perf_counter() - start_t
            
            print(f"  Frame {frame_index} prediction time: {pred_time:.5f} seconds")
            total_time += pred_time
            
            with open(out_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([video_id, qp, recon_yuv_path.name, frame_index, pred_time])
                        
print(f"\nTotal prediction time for all measured frames: {total_time:.5f} seconds")
