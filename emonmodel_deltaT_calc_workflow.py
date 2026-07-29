# %% [markdown]
# # My Own Model — ConvLSTM Next-Frame Predictor
# 
# ConvLSTM-based next-frame predictor for Inter4K GOP-8 Y-channel frame prediction. It keeps the same input/output format as `model2.ipynb`: `(B, 7, 1, H, W) -> (B, 1, H, W)`.

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
from pathlib import Path

ON_KAGGLE = os.path.exists('/kaggle/working')
ON_GITHUB = os.environ.get('GITHUB_ACTIONS') == 'true'

if ON_KAGGLE:
    DATASET_ROOT = '/kaggle/input/datasets/tonmoyk983/sevtone-4-qp-gop8/sevtone_4_QP_GOP8/Inter4K/RAW'
    CKPT_DIR = '/kaggle/working'
else:
    # Use relative paths for local and GitHub Actions
    DATASET_ROOT = './Inter4K/RAW'
    CKPT_DIR = '.'

CKPT_DIR = Path(CKPT_DIR)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CKPT_DIR / 'best_model.pt'
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

env_name = "Kaggle" if ON_KAGGLE else ("GitHub Actions" if ON_GITHUB else "Local")
print(f'Running on: {env_name}')
print(f'DATASET_ROOT : {DATASET_ROOT}')
print(f'Device       : {DEVICE}')
print(f'Best model   : {BEST_MODEL_PATH}')



# %% [markdown]
# ## My Model — ConvLSTM Predictor
# 
# Stacked ConvLSTM encoder (with spatial downsampling) followed by a pixel-shuffle upsample + conv refinement head. Consumes the full `T`-frame sequence recurrently and predicts the next frame from the final hidden state.

# %%
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=bias
        )

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class ConvLSTMEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, kernel_size=3, downsample=2,
                 use_checkpoint=False):
        super().__init__()
        self.downsample = downsample
        self.use_checkpoint = use_checkpoint

        if downsample > 1:
            self.ds_conv  = nn.Conv2d(in_channels, hidden_dims[0],
                                      kernel_size=3, stride=downsample, padding=1)
            cell_in_chs   = [hidden_dims[0]] + hidden_dims[:-1]
        else:
            self.ds_conv  = nn.Identity()
            cell_in_chs   = [in_channels] + hidden_dims[:-1]

        self.cells = nn.ModuleList([
            ConvLSTMCell(cell_in_chs[i], hidden_dims[i], kernel_size)
            for i in range(len(hidden_dims))
        ])

    def _step(self, x_t, *flat_states):
        states = [(flat_states[2 * i], flat_states[2 * i + 1])
                  for i in range(len(self.cells))]
        layer_input = self.ds_conv(x_t)
        new_states = []
        for cell, (h, c) in zip(self.cells, states):
            h, c = cell(layer_input, (h, c))
            new_states.append((h, c))
            layer_input = h
        flat_out = []
        for h, c in new_states:
            flat_out.extend([h, c])
        return tuple(flat_out)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        fH = H // self.downsample if self.downsample > 1 else H
        fW = W // self.downsample if self.downsample > 1 else W
        states = [cell.init_hidden(B, fH, fW, x_seq.device) for cell in self.cells]
        flat_states = tuple(t for hc in states for t in hc)

        for t in range(T):
            x_t = x_seq[:, t]
            if self.use_checkpoint and self.training:
                flat_states = grad_checkpoint(
                    self._step, x_t, *flat_states, use_reentrant=False
                )
            else:
                flat_states = self._step(x_t, *flat_states)

        n = len(self.cells)
        h_list = [flat_states[2 * i] for i in range(n)]
        c_list = [flat_states[2 * i + 1] for i in range(n)]
        return h_list, c_list


class PredictionHead(nn.Module):
    """Pixel-shuffle upsample + conv refinement head."""

    def __init__(self, in_channels, out_channels=3, upsample=2):
        super().__init__()
        if upsample > 1:
            self.up   = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * (upsample ** 2), 3, padding=1),
                nn.PixelShuffle(upsample),
            )
            refine_in = in_channels
        else:
            self.up   = nn.Identity()
            refine_in = in_channels

        self.refine = nn.Sequential(
            nn.Conv2d(refine_in, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),        nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, 1),
            nn.Tanh(),
        )

    def forward(self, h):
        return self.refine(self.up(h))


class ConvLSTMPredictor(nn.Module):
    """End-to-end ConvLSTM next-frame predictor."""

    def __init__(self, in_channels=1, hidden_dims=[64, 128, 64],
                 kernel_size=3, downsample=2, use_checkpoint=False):
        super().__init__()
        self.encoder = ConvLSTMEncoder(in_channels, hidden_dims, kernel_size,
                                        downsample, use_checkpoint=use_checkpoint)
        self.head    = PredictionHead(hidden_dims[-1], in_channels, downsample)

    def forward(self, x_seq):
        h_last, _ = self.encoder(x_seq)
        return self.head(h_last[-1])


# ConvLSTM architecture config (matches training notebook: SEQ_LEN=7, IN_CHANNELS=1)
HIDDEN_DIMS = [64, 128, 64]
KERNEL_SIZE = 3
DOWNSAMPLE = 2

model = ConvLSTMPredictor(
    in_channels=1, hidden_dims=HIDDEN_DIMS,
    kernel_size=KERNEL_SIZE, downsample=DOWNSAMPLE,
    use_checkpoint=False,
)
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
EVAL_MODEL_PATH = BEST_MODEL_PATH  # convLSTM_final.pt

if EVAL_MODEL_PATH.exists():
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    ckpt = torch.load(EVAL_MODEL_PATH, map_location=DEVICE)
    # The training notebook saves {'epoch':..., 'state_dict':..., ...}
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    base_model.load_state_dict(state_dict)
    print(f'Loaded checkpoint: {EVAL_MODEL_PATH}')
else:
    print('No checkpoint found. Running with untrained weights.')

model.eval()

# You can change this to point to the root containing all class folders.
TESTING_CLASSD2_ROOT = Path('segments/Testing_ClassD')
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
