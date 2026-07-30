# =============================================================================
# Step 1: Model Prediction Time Measurement Script
# =============================================================================
#
# Measures inference execution time of the ConvLSTM-UNet model (EnhancedMotionPredNet)
# for the 4 target GOP-8 last frames (indices 8, 16, 24, 32) across all test videos & QPs.
#
# Outputs:
#   prediction_times.csv  — Contains (testing_class, video_id, qp, frame_index, pred_time_sec)
#                           to be consumed by Step 2 (delta_t_calculate.py).
#
# Usage:
#   python calculate_prediction_time.py
# =============================================================================

import os
import re
import time
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# =============================================================================
# CONFIGURATION
# =============================================================================

# Root directory containing Testing_ClassA .. Testing_ClassD subdirectories
TESTING_ROOT = r"D:\Dataset\Inter4K\60fps\UHD\Segments\sevtone_4_QP_GOP8\Reconstructed"

# Path to trained model checkpoint
CKPT_PATH = r"Taki_model.pt"

# Output directory & output CSV path
OUT_DIR  = r"D:\Dataset\Inter4K\60fps\UHD\Segments\sevtone_4_QP_GOP8\Reconstructed"
PRED_CSV = os.path.join(OUT_DIR, "prediction_times.csv")

# GOP configuration
GOP_SIZE      = 8
SEQ_LEN       = GOP_SIZE - 1    # 7 input frames
TARGET_FRAMES = [8, 16, 24, 32] # 1-indexed frame numbers (last frame of each GOP-8)
QPS           = ["QP_37", "QP_42", "QP_47", "QP_51"]

CLASS_FOLDERS = ["Testing_ClassA"]

# Benchmark timing settings
WARMUP_ITERS = 2
TIMED_ITERS  = 5


# =============================================================================
# DEVICE SETUP
# =============================================================================

def resolve_device():
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        major, _minor = torch.cuda.get_device_capability(0)
        if major < 7:
            return 'cpu'
    except Exception:
        return 'cpu'
    return 'cuda'

DEVICE = resolve_device()
USE_AMP = True
amp_enabled = USE_AMP and (DEVICE == 'cuda')


# =============================================================================
# MODEL ARCHITECTURE (EnhancedMotionPredNet)
# =============================================================================

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell operating on spatial feature maps."""
    def __init__(self, in_channels, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=bias
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device, dtype=torch.float32):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype),
        )


class AttnResBlock(nn.Module):
    """Residual Block with GroupNorm and Squeeze-and-Excitation Motion Attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        groups = min(8, channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        res = self.conv(x)
        res = res * self.se(res)
        return self.act(x + res)


class EnhancedMotionEncoder(nn.Module):
    """3-Stage Feature Pyramid Encoder accepting 2-channel input."""
    def __init__(self, in_channels=2, feat_dim=64):
        super().__init__()
        self.s1_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.s1_res = nn.Sequential(AttnResBlock(32), AttnResBlock(32))
        self.s2_ds = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.s2_res = nn.Sequential(AttnResBlock(64), AttnResBlock(64), AttnResBlock(64))
        self.s3_ds = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.s3_res = nn.Sequential(AttnResBlock(128), AttnResBlock(128), AttnResBlock(128))
        self.s3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(64 + 64, feat_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x):
        f1 = self.s1_res(self.s1_stem(x))
        f2 = self.s2_res(self.s2_ds(f1))
        f3 = self.s3_res(self.s3_ds(f2))
        f3_up = self.s3_up(f3)
        fused = self.fuse_conv(torch.cat([f2, f3_up], dim=1))
        return f1, fused


class EnhancedMotionPredNet(nn.Module):
    """Enhanced Motion Predictor with Dual-Scale Skip Connections & ConvLSTM Bottleneck."""
    def __init__(self, in_channels=1, hidden_dims=None, kernel_size=3, downsample=2, use_checkpoint=False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 64]
        self.downsample = downsample
        self.use_checkpoint = use_checkpoint

        self.encoder = EnhancedMotionEncoder(in_channels=2, feat_dim=hidden_dims[0])

        cell_in_chs = [hidden_dims[0]] + hidden_dims[:-1]
        self.cells = nn.ModuleList([
            ConvLSTMCell(cell_in_chs[i], hidden_dims[i], kernel_size)
            for i in range(len(hidden_dims))
        ])

        self.mid_fuse = nn.Sequential(
            nn.Conv2d(hidden_dims[-1] + hidden_dims[0], hidden_dims[-1], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.up = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1] * (downsample ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(downsample),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dims[-1] + 32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            AttnResBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.Tanh()
        )

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        fH, fW = H // self.downsample, W // self.downsample

        device = x_seq.device
        dtype = x_seq.dtype
        states = [cell.init_hidden(B, fH, fW, device, dtype) for cell in self.cells]

        f1_last = None
        fused_last = None

        for t in range(T):
            frame = x_seq[:, t]
            prev_frame = x_seq[:, t - 1] if t > 0 else frame
            velocity_map = frame - prev_frame
            enc_input = torch.cat([frame, velocity_map], dim=1)

            f1, fused_feat = self.encoder(enc_input)

            if t == T - 1:
                f1_last = f1
                fused_last = fused_feat

            new_states = []
            layer_in = fused_feat
            for cell, state in zip(self.cells, states):
                h, c = cell(layer_in, state)
                new_states.append((h, c))
                layer_in = h
            states = new_states

        h_last = states[-1][0]
        h_fused = self.mid_fuse(torch.cat([h_last, fused_last], dim=1))

        up_feat = self.up(h_fused)
        fused = torch.cat([up_feat, f1_last], dim=1)
        delta = self.refine(fused)

        pred = x_seq[:, -1] + self.res_scale * delta
        return pred.clamp(-1.0, 1.0)


def get_module(m):
    """Unwrap DataParallel if active."""
    return m.module if isinstance(m, nn.DataParallel) else m


# =============================================================================
# NORMALIZATION & YUV HELPERS
# =============================================================================

_NORM_MEAN = [0.5]
_NORM_STD  = [0.5]

def read_yuv_frame(yuv_path, width, height, frame_idx):
    """Read a single Y-channel frame from a raw YUV420 file as a PIL Image."""
    n_y = width * height
    n_uv = (width // 2) * (height // 2) * 2
    frame_sz = n_y + n_uv

    with open(yuv_path, 'rb') as f:
        f.seek(frame_idx * frame_sz)
        raw = f.read(frame_sz)

    if len(raw) < frame_sz:
        raise ValueError(f"Frame {frame_idx}: expected {frame_sz} bytes, got {len(raw)}.")

    y_plane = np.frombuffer(raw[:n_y], dtype=np.uint8).reshape(height, width)
    return Image.fromarray(y_plane, mode='L')


def parse_resolution(video_name):
    """Extract (width, height) from video name like 'BasketballPass_416x240_50'."""
    m = re.search(r'_(\d+)x(\d+)_', video_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot parse resolution from video name: {video_name}")


# =============================================================================
# TIMING BENCHMARK
# =============================================================================

def measure_prediction_time(model, vvc_yuv_path, width, height, gop_start_0idx, model_resize):
    """
    Measures average inference time for predicting the last frame of one GOP.
    """
    to_model_res = transforms.Compose([
        transforms.Resize(model_resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
    ])

    # Read preceding 7 VVC frames
    input_pils = [read_yuv_frame(vvc_yuv_path, width, height, gop_start_0idx + i) for i in range(SEQ_LEN)]
    input_tensors = [to_model_res(pil).unsqueeze(0).to(DEVICE) for pil in input_pils]
    x = torch.stack(input_tensors, dim=1)  # (1, T=7, C=1, 512, 512)

    # Warmup runs
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(x)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(TIMED_ITERS):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    return float(np.mean(times))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print(f"[Step 1] Loading model checkpoint from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    _ckpt_cfg = ckpt.get('config', {})

    model = EnhancedMotionPredNet(
        in_channels=_ckpt_cfg.get('in_channels', 1),
        hidden_dims=_ckpt_cfg.get('hidden_dims', [64, 128, 64]),
        downsample=_ckpt_cfg.get('downsample', 2),
        use_checkpoint=False
    ).to(DEVICE)

    get_module(model).load_state_dict(ckpt['state_dict'])
    model.eval()

    _model_frame_size = _ckpt_cfg.get('frame_size', 512)
    _MODEL_RESIZE = (_model_frame_size, _model_frame_size)

    print(f"[Step 1] Device: {DEVICE}  | AMP: {amp_enabled}  | Resize: {_MODEL_RESIZE}")
    print(f"[Step 1] Measuring model prediction times across test sequences...\n")

    records = []

    for class_folder in CLASS_FOLDERS:
        class_dir = os.path.join(TESTING_ROOT, class_folder)
        if not os.path.isdir(class_dir):
            continue

        videos = sorted([d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))])

        for video in videos:
            video_dir = os.path.join(class_dir, video)
            width, height = parse_resolution(video)

            for qp in QPS:
                qp_dir = os.path.join(video_dir, qp)
                if not os.path.isdir(qp_dir):
                    continue

                # Locate reconstructed YUV
                vvc_yuv = None
                for fname in os.listdir(qp_dir):
                    if fname.startswith("C_") and fname.endswith(".yuv"):
                        vvc_yuv = os.path.join(qp_dir, fname)
                        break

                if not vvc_yuv or not os.path.exists(vvc_yuv):
                    continue

                print(f"  Measuring [{class_folder}] {video} / {qp} ...", end=" ", flush=True)

                for frame_idx in TARGET_FRAMES:
                    gop_start_0idx = frame_idx - GOP_SIZE
                    try:
                        pred_sec = measure_prediction_time(model, vvc_yuv, width, height, gop_start_0idx, _MODEL_RESIZE)
                    except Exception as e:
                        print(f"\n    [Error] Frame {frame_idx}: {e}")
                        pred_sec = 0.0

                    records.append({
                        "testing_class": class_folder,
                        "video_id": video,
                        "qp": qp,
                        "frame_index": frame_idx,
                        "pred_time_sec": round(pred_sec, 6)
                    })

                print("Done.")

    os.makedirs(OUT_DIR, exist_ok=True)
    fieldnames = ["testing_class", "video_id", "qp", "frame_index", "pred_time_sec"]

    with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[Step 1 COMPLETE] Saved {len(records)} prediction time records to:")
    print(f"  {PRED_CSV}")


if __name__ == "__main__":
    main()
