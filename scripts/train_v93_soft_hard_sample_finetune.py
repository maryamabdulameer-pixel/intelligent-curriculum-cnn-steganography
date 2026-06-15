# ============================================================
# V9.3 - SOFT HARD-SAMPLE FINE-TUNING
# Single Cell - Starts from Best V9.1
# Uses hard samples mined by V9.2, but softly
# Early stopping patience = 10
# FIXED PATHS: /content/gdrive/MyDrive
# ============================================================

import os, glob, json, time, random, warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

warnings.filterwarnings("ignore")

# ============================================================
# 1) SETUP + PATHS
# ============================================================

from google.colab import drive

if not os.path.exists("/content/gdrive/MyDrive"):
    drive.mount("/content/gdrive")
else:
    print("Google Drive already mounted at /content/gdrive")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True if DEVICE == "cuda" else False

print("="*100)
print("V9.3 - SOFT HARD-SAMPLE FINE-TUNING")
print("="*100)
print("DEVICE:", DEVICE)

MYDRIVE = "/content/gdrive/MyDrive"
PROJECT_ROOT = f"{MYDRIVE}/stego_cnn_project"

PAIR_ROOT = f"{PROJECT_ROOT}/coco_20k_stego_pairs"

TRAIN_COVERS = f"{PAIR_ROOT}/train/covers"
TRAIN_SECRETS = f"{PAIR_ROOT}/train/secrets"
VAL_COVERS   = f"{PAIR_ROOT}/val/covers"
VAL_SECRETS  = f"{PAIR_ROOT}/val/secrets"
TEST_COVERS  = f"{PAIR_ROOT}/test/covers"
TEST_SECRETS = f"{PAIR_ROOT}/test/secrets"

V91_BEST_CKPT = f"{PROJECT_ROOT}/runs_V91_hard_negative_security_finetune/checkpoints/best_v91_hard_negative.pt"
V91_LAST_CKPT = f"{PROJECT_ROOT}/runs_V91_hard_negative_security_finetune/checkpoints/last_v91_hard_negative.pt"

V92_HARD_JSON = f"{PROJECT_ROOT}/runs_V92_hard_sample_mining_finetune/hard_samples/train_hard_indices.json"
V92_HARD_CSV  = f"{PROJECT_ROOT}/runs_V92_hard_sample_mining_finetune/hard_samples/train_hard_mining_results.csv"

RUN_DIR = f"{PROJECT_ROOT}/runs_V93_soft_hard_sample_finetune"

for sub in ["checkpoints", "best", "samples", "metrics", "config", "hard_samples"]:
    os.makedirs(f"{RUN_DIR}/{sub}", exist_ok=True)

if os.path.exists(V91_BEST_CKPT):
    START_CKPT = V91_BEST_CKPT
elif os.path.exists(V91_LAST_CKPT):
    START_CKPT = V91_LAST_CKPT
else:
    raise FileNotFoundError("No V9.1 checkpoint found. شغلي V9.1 أولاً.")

print("PROJECT_ROOT:", PROJECT_ROOT)
print("PAIR_ROOT:", PAIR_ROOT)
print("START_CKPT:", START_CKPT)
print("V92_HARD_JSON:", V92_HARD_JSON)
print("RUN_DIR:", RUN_DIR)

# ============================================================
# 2) HYPERPARAMETERS - SOFT VERSION
# ============================================================

IMG_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 2

EPOCHS = 30
SAVE_EVERY = 5
PATIENCE = 10

LR_E  = 4.0e-6
LR_R  = 6.0e-6
LR_DX = 2.0e-5
LR_DS = 1.5e-5

ALPHA_MIN = 0.054
ALPHA_MAX = 0.060

LAMBDA_COVER = 8.5
LAMBDA_SECRET_RGB = 6.8
LAMBDA_SECRET_Y = 9.5
LAMBDA_SECRET_COLOR = 2.0

LAMBDA_EDGE = 1.2
LAMBDA_FFT = 2.5
LAMBDA_RESIDUAL = 2.8
LAMBDA_COLOR_STAT = 1.0
LAMBDA_MASK_SMOOTH = 0.45
LAMBDA_MASK_MEAN = 0.25
TARGET_MASK_MEAN = 0.72

# Soft detector pressure
LAMBDA_OLD_XUNET_MAX = 0.030
LAMBDA_NEW_SRNET_MAX = 0.012

ADV_WARMUP_EPOCHS = 5

SECRET_Y_GATE = 30.80
BER_GATE = 0.0355

MIN_COVER_Y = 63.0
MIN_SECRET_Y = 30.80
MAX_BER = 0.0355
MAX_AVG_DET = 0.060

HARD_THRESHOLD_X = 0.50
HARD_THRESHOLD_S = 0.50

# Soft hard sampling
HARD_SAMPLE_WEIGHT = 2.0
NORMAL_SAMPLE_WEIGHT = 1.0

# لو تريدين تعيدين استخراج الصور الصعبة خليها True
REBUILD_HARD_MINING = False

config = {
    "version": "V9.3 Soft Hard-Sample Fine-Tuning",
    "start_from": START_CKPT,
    "run_dir": RUN_DIR,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "hard_sample_weight": HARD_SAMPLE_WEIGHT,
    "normal_sample_weight": NORMAL_SAMPLE_WEIGHT,
    "lambda_old_xunet_max": LAMBDA_OLD_XUNET_MAX,
    "lambda_new_srnet_max": LAMBDA_NEW_SRNET_MAX,
    "goal": "Use hard samples softly to improve security without damaging recovery."
}

with open(f"{RUN_DIR}/config/config_v93.json", "w") as f:
    json.dump(config, f, indent=4)

# ============================================================
# 3) DATASET
# ============================================================

def list_images(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
            "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

class StegoPairsDataset(Dataset):
    def __init__(self, covers_dir, secrets_dir, img_size=256):
        self.covers = list_images(covers_dir)
        self.secrets = list_images(secrets_dir)

        n = min(len(self.covers), len(self.secrets))
        self.covers = self.covers[:n]
        self.secrets = self.secrets[:n]

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        print(f"Dataset: {covers_dir}")
        print(f"covers={len(self.covers)}, secrets={len(self.secrets)}")

        if len(self.covers) == 0 or len(self.secrets) == 0:
            raise ValueError(f"Dataset empty. Check: {covers_dir} and {secrets_dir}")

    def __len__(self):
        return len(self.covers)

    def __getitem__(self, idx):
        cover = Image.open(self.covers[idx]).convert("RGB")
        secret = Image.open(self.secrets[idx]).convert("RGB")
        return self.tf(cover), self.tf(secret), idx

train_ds = StegoPairsDataset(TRAIN_COVERS, TRAIN_SECRETS, IMG_SIZE)
val_ds   = StegoPairsDataset(VAL_COVERS, VAL_SECRETS, IMG_SIZE)
test_ds  = StegoPairsDataset(TEST_COVERS, TEST_SECRETS, IMG_SIZE)

base_train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("Train samples:", len(train_ds))
print("Val samples  :", len(val_ds))
print("Test samples :", len(test_ds))

# ============================================================
# 4) METRICS + LOSSES
# ============================================================

def rgb_to_y(x):
    return 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]

def rgb_to_cbcr(x):
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    cb = -0.168736*r - 0.331264*g + 0.5*b + 0.5
    cr = 0.5*r - 0.418688*g - 0.081312*b + 0.5
    return torch.cat([cb, cr], dim=1).clamp(0, 1)

def batch_psnr(x, y, eps=1e-8):
    mse = torch.mean((x-y)**2, dim=[1,2,3])
    return (10 * torch.log10(1.0 / (mse + eps))).mean()

def simple_ssim(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    sig_x = F.avg_pool2d(x*x, 11, 1, 5) - mu_x*mu_x
    sig_y = F.avg_pool2d(y*y, 11, 1, 5) - mu_y*mu_y
    sig_xy = F.avg_pool2d(x*y, 11, 1, 5) - mu_x*mu_y
    ssim = ((2*mu_x*mu_y+C1)*(2*sig_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sig_x+sig_y+C2))
    return ssim.mean()

def ncc(x, y, eps=1e-8):
    x = x - x.mean(dim=[1,2,3], keepdim=True)
    y = y - y.mean(dim=[1,2,3], keepdim=True)
    num = torch.sum(x*y, dim=[1,2,3])
    den = torch.sqrt(torch.sum(x*x, dim=[1,2,3]) * torch.sum(y*y, dim=[1,2,3]) + eps)
    return (num / den).mean()

def ber_binary(secret, recovered):
    s = (secret > 0.5).float()
    r = (recovered > 0.5).float()
    return (s != r).float().mean()

def total_variation_loss(x):
    dh = torch.mean(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]))
    dw = torch.mean(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]))
    return dh + dw

def sobel_edges_y(x):
    y = rgb_to_y(x)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-8)

def edge_loss(cover, stego):
    return F.l1_loss(sobel_edges_y(cover), sobel_edges_y(stego))

def fft_band_loss(cover, stego):
    cy = rgb_to_y(cover)
    sy = rgb_to_y(stego)

    cf = torch.fft.fftshift(torch.fft.fft2(cy, norm="ortho"), dim=(-2,-1))
    sf = torch.fft.fftshift(torch.fft.fft2(sy, norm="ortho"), dim=(-2,-1))

    cm = torch.log1p(torch.abs(cf))
    sm = torch.log1p(torch.abs(sf))

    b, c, h, w = cm.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=cover.device),
        torch.arange(w, device=cover.device),
        indexing="ij"
    )

    rr = torch.sqrt((yy-h//2)**2 + (xx-w//2)**2)
    rr = rr / (rr.max() + 1e-8)

    bands = [(0.0,0.12), (0.12,0.30), (0.30,0.55), (0.55,1.0)]
    loss = 0.0

    for lo, hi in bands:
        m = ((rr >= lo) & (rr < hi)).float().view(1,1,h,w)
        cmv = (cm*m).sum(dim=[2,3]) / (m.sum()+1e-8)
        smv = (sm*m).sum(dim=[2,3]) / (m.sum()+1e-8)
        loss += F.l1_loss(smv, cmv)

    return loss / len(bands)

def residual_distribution_loss(cover, stego):
    res = stego - cover
    return torch.abs(res.mean()) + 0.5*res.std() + 0.5*total_variation_loss(res)

def color_stat_loss(cover, stego):
    cy, sy = rgb_to_y(cover), rgb_to_y(stego)
    cc, sc = rgb_to_cbcr(cover), rgb_to_cbcr(stego)

    return (
        F.l1_loss(sy.mean(dim=[2,3]), cy.mean(dim=[2,3])) +
        F.l1_loss(sy.std(dim=[2,3]),  cy.std(dim=[2,3])) +
        F.l1_loss(sc.mean(dim=[2,3]), cc.mean(dim=[2,3])) +
        F.l1_loss(sc.std(dim=[2,3]),  cc.std(dim=[2,3]))
    )

def mask_regularization(mask):
    return LAMBDA_MASK_SMOOTH * total_variation_loss(mask) + LAMBDA_MASK_MEAN * torch.abs(mask.mean() - TARGET_MASK_MEAN)

def texture_guided_mask_loss(cover, mask):
    edges = sobel_edges_y(cover)
    edges = edges / (edges.amax(dim=[2,3], keepdim=True) + 1e-8)
    target = 0.35 + 0.65 * edges.detach()
    return F.l1_loss(mask, target)

def secret_loss(secret, recovered):
    sy, ry = rgb_to_y(secret), rgb_to_y(recovered)
    sc, rc = rgb_to_cbcr(secret), rgb_to_cbcr(recovered)

    loss_rgb = F.mse_loss(recovered, secret)
    loss_y = F.mse_loss(ry, sy)
    loss_color = F.mse_loss(rc, sc)

    total = (
        LAMBDA_SECRET_RGB * loss_rgb +
        LAMBDA_SECRET_Y * loss_y +
        LAMBDA_SECRET_COLOR * loss_color
    )

    return total, loss_rgb, loss_y, loss_color

# ============================================================
# 5) MODELS
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        if norm:
            layers.insert(1, nn.BatchNorm2d(out_ch))
            layers.insert(4, nn.BatchNorm2d(out_ch))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class HidingBackbone(nn.Module):
    def __init__(self, base=48):
        super().__init__()

        self.enc1 = ConvBlock(6, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base*4, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, base)

        self.out = nn.Conv2d(base, 3, 3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(0.0595))

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.mid(e3)

        d2 = self.up2(m)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        residual = torch.tanh(self.out(d1))
        alpha = torch.clamp(self.alpha, ALPHA_MIN, ALPHA_MAX)

        return residual, alpha

class MaskNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(3, base),
            nn.MaxPool2d(2),
            ConvBlock(base, base*2),
            nn.MaxPool2d(2),
            ConvBlock(base*2, base*4),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base*4, base*2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base*2, base),
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, cover):
        raw = self.net(cover)
        return 0.35 + 0.65 * raw

class EmbeddingNetwork(nn.Module):
    def __init__(self, base=48):
        super().__init__()

        self.hider = HidingBackbone(base=base)
        self.masker = MaskNet(base=24)

    def forward(self, cover, secret):
        mask = self.masker(cover)
        residual, alpha = self.hider(cover, secret)
        stego = cover + alpha * mask * residual
        stego = torch.clamp(stego, 0.0, 1.0)

        return stego, residual, mask, alpha

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()

        hidden = max(ch // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class RecoveryNetwork(nn.Module):
    def __init__(self, base=48):
        super().__init__()

        self.block1 = ConvBlock(3, base)
        self.block2 = ConvBlock(base, base*2)
        self.pool = nn.MaxPool2d(2)
        self.block3 = ConvBlock(base*2, base*4)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block4 = ConvBlock(base*4, base*2)
        self.se = SEBlock(base*2)
        self.block5 = ConvBlock(base*2, base)
        self.out = nn.Conv2d(base, 3, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, stego):
        x = self.block1(stego)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.up(x)
        x = self.block4(x)
        x = self.se(x)
        x = self.block5(x)

        return self.sigmoid(self.out(x))

class XuNetStyleDetector(nn.Module):
    def __init__(self, base=32):
        super().__init__()

        self.high_pass = nn.Conv2d(3, 3, 5, padding=2, bias=False)

        hp = torch.tensor([
            [ 0,  0, -1,  0,  0],
            [ 0, -1,  2, -1,  0],
            [-1,  2,  4,  2, -1],
            [ 0, -1,  2, -1,  0],
            [ 0,  0, -1,  0,  0]
        ], dtype=torch.float32) / 4.0

        with torch.no_grad():
            w = torch.zeros(3, 3, 5, 5)
            for c in range(3):
                w[c, c] = hp
            self.high_pass.weight.copy_(w)

        for p in self.high_pass.parameters():
            p.requires_grad = False

        self.features = nn.Sequential(
            nn.Conv2d(3, base, 5, padding=2),
            nn.BatchNorm2d(base),
            nn.Tanh(),
            nn.AvgPool2d(2),

            nn.Conv2d(base, base*2, 5, padding=2),
            nn.BatchNorm2d(base*2),
            nn.Tanh(),
            nn.AvgPool2d(2),

            nn.Conv2d(base*2, base*4, 3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(base*4, base*8, 3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base*8, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(self.high_pass(x)))

class SRNetStyleDetector(nn.Module):
    def __init__(self, base=32):
        super().__init__()

        self.high_pass = nn.Conv2d(3, 3, 5, padding=2, bias=False)

        hp = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32) / 12.0

        with torch.no_grad():
            w = torch.zeros(3, 3, 5, 5)
            for c in range(3):
                w[c, c] = hp
            self.high_pass.weight.copy_(w)

        for p in self.high_pass.parameters():
            p.requires_grad = False

        self.features = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(base, base*2, 3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base*2, base*2, 3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(base*2, base*4, 3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base*4, base*4, 3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(base*4, base*8, 3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base*8, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(self.high_pass(x)))

def set_detector_trainable(D, flag=True):
    for p in D.parameters():
        p.requires_grad = flag
    for p in D.high_pass.parameters():
        p.requires_grad = False

# ============================================================
# 6) INIT MODELS
# ============================================================

E = EmbeddingNetwork(base=48).to(DEVICE)
R = RecoveryNetwork(base=48).to(DEVICE)
DX = XuNetStyleDetector(base=32).to(DEVICE)
DS = SRNetStyleDetector(base=32).to(DEVICE)

bce = nn.BCEWithLogitsLoss()

opt_E = torch.optim.AdamW(E.parameters(), lr=LR_E, betas=(0.5, 0.999), weight_decay=1e-5)
opt_R = torch.optim.AdamW(R.parameters(), lr=LR_R, betas=(0.5, 0.999), weight_decay=1e-5)
opt_DX = torch.optim.AdamW([p for p in DX.parameters() if p.requires_grad], lr=LR_DX, betas=(0.5, 0.999), weight_decay=1e-5)
opt_DS = torch.optim.AdamW([p for p in DS.parameters() if p.requires_grad], lr=LR_DS, betas=(0.5, 0.999), weight_decay=1e-5)

sched_E  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_E, T_max=EPOCHS, eta_min=1e-6)
sched_R  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_R, T_max=EPOCHS, eta_min=1e-6)
sched_DX = torch.optim.lr_scheduler.CosineAnnealingLR(opt_DX, T_max=EPOCHS, eta_min=1e-6)
sched_DS = torch.optim.lr_scheduler.CosineAnnealingLR(opt_DS, T_max=EPOCHS, eta_min=1e-6)

scaler_E  = torch.amp.GradScaler("cuda", enabled=USE_AMP)
scaler_R  = torch.amp.GradScaler("cuda", enabled=USE_AMP)
scaler_DX = torch.amp.GradScaler("cuda", enabled=USE_AMP)
scaler_DS = torch.amp.GradScaler("cuda", enabled=USE_AMP)

best_score = -1e9
start_epoch = 1
no_improve = 0

best_path = f"{RUN_DIR}/checkpoints/best_v93_soft_hard_sample.pt"
last_path = f"{RUN_DIR}/checkpoints/last_v93_soft_hard_sample.pt"

if os.path.exists(last_path):
    print("Found V9.3 last checkpoint. Resuming...")

    ckpt = torch.load(last_path, map_location=DEVICE)

    E.load_state_dict(ckpt["E"])
    R.load_state_dict(ckpt["R"])
    DX.load_state_dict(ckpt["DX"])
    DS.load_state_dict(ckpt["DS"])

    opt_E.load_state_dict(ckpt["opt_E"])
    opt_R.load_state_dict(ckpt["opt_R"])
    opt_DX.load_state_dict(ckpt["opt_DX"])
    opt_DS.load_state_dict(ckpt["opt_DS"])

    sched_E.load_state_dict(ckpt["sched_E"])
    sched_R.load_state_dict(ckpt["sched_R"])
    sched_DX.load_state_dict(ckpt["sched_DX"])
    sched_DS.load_state_dict(ckpt["sched_DS"])

    best_score = ckpt.get("best_score", -1e9)
    start_epoch = ckpt["epoch"] + 1
    no_improve = ckpt.get("no_improve", 0)

    print("Resumed from epoch:", ckpt["epoch"])
    print("Current best score:", best_score)

else:
    print("Starting new V9.3 from best V9.1 checkpoint...")

    ckpt = torch.load(START_CKPT, map_location=DEVICE)

    E.load_state_dict(ckpt["E"])
    R.load_state_dict(ckpt["R"])
    DX.load_state_dict(ckpt["DX"])
    DS.load_state_dict(ckpt["DS"])

    print("Loaded best V9.1.")
    print("Best V9.1 epoch:", ckpt.get("epoch", "unknown"))
    print("Best V9.1 score:", ckpt.get("best_score", "unknown"))

# ============================================================
# 7) PANEL SAVE
# ============================================================

def tensor_rgb_to_np(x):
    return x.detach().float().cpu().clamp(0,1).permute(1,2,0).numpy().astype(np.float32)

def tensor_mask_to_np(x):
    return x.detach().float().cpu().clamp(0,1).numpy().astype(np.float32)

def save_titled_panel(cover, secret, stego, recovered, mask, path, epoch, split="val", n=4):
    n = min(n, cover.size(0))

    fig, axes = plt.subplots(n, 5, figsize=(18, 4*n))
    titles = ["Cover", "Secret", "Stego", "Recovered", "Mask"]

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        imgs = [
            tensor_rgb_to_np(cover[i]),
            tensor_rgb_to_np(secret[i]),
            tensor_rgb_to_np(stego[i]),
            tensor_rgb_to_np(recovered[i]),
            tensor_mask_to_np(mask[i,0])
        ]

        for j in range(5):
            ax = axes[i, j]
            if j == 4:
                ax.imshow(imgs[j], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(imgs[j])
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[j], fontsize=15, fontweight="bold")

    fig.suptitle(
        f"V9.3 Best Epoch {epoch} - {split.upper()} | Cover | Secret | Stego | Recovered | Mask",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

# ============================================================
# 8) HARD SAMPLE LOADING / MINING
# ============================================================

@torch.no_grad()
def mine_hard_samples(loader, split_name="train"):
    E.eval()
    R.eval()
    DX.eval()
    DS.eval()

    rows = []
    hard_indices = []

    print("\n" + "="*100)
    print(f"MINING HARD SAMPLES FROM {split_name.upper()}")
    print("="*100)

    for cover, secret, idx in tqdm(loader, desc=f"Mining hard samples: {split_name}"):
        cover = cover.to(DEVICE, non_blocking=True)
        secret = secret.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            stego, residual, mask, alpha = E(cover, secret)
            recovered = R(stego)

            px_s = torch.sigmoid(DX(stego)).view(-1)
            ps_s = torch.sigmoid(DS(stego)).view(-1)

            det_x = px_s >= HARD_THRESHOLD_X
            det_s = ps_s >= HARD_THRESHOLD_S
            det_any = det_x | det_s

            sec_y = rgb_to_y(secret)
            rec_y = rgb_to_y(recovered)

            mse_y = torch.mean((rec_y - sec_y) ** 2, dim=[1,2,3])
            psnr_y = 10 * torch.log10(1.0 / (mse_y + 1e-8))

            s_bin = (secret > 0.5).float()
            r_bin = (recovered > 0.5).float()
            ber_per = (s_bin != r_bin).float().mean(dim=[1,2,3])

        for i in range(cover.size(0)):
            sample_idx = int(idx[i].item())
            is_hard = int(det_any[i].item())

            if is_hard:
                hard_indices.append(sample_idx)

            rows.append({
                "idx": sample_idx,
                "xunet_prob_stego": float(px_s[i].item()),
                "srnet_prob_stego": float(ps_s[i].item()),
                "xunet_detected": int(det_x[i].item()),
                "srnet_detected": int(det_s[i].item()),
                "detected_any": is_hard,
                "secret_psnr_y": float(psnr_y[i].item()),
                "ber_secret": float(ber_per[i].item()),
                "mask_mean": float(mask[i].mean().item()),
                "alpha": float(alpha.detach().cpu())
            })

    df = pd.DataFrame(rows)
    hard_indices = sorted(list(set(hard_indices)))

    csv_path = f"{RUN_DIR}/hard_samples/train_hard_mining_results_v93.csv"
    json_path = f"{RUN_DIR}/hard_samples/train_hard_indices_v93.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(hard_indices, f, indent=4)

    print("\nHard mining saved:")
    print(csv_path)
    print(json_path)
    print("Total samples:", len(df))
    print("Hard samples :", len(hard_indices))
    print("Hard ratio   :", len(hard_indices) / max(1, len(df)))

    return hard_indices, df

# Load V9.2 hard indices if available
if os.path.exists(V92_HARD_JSON) and not REBUILD_HARD_MINING:
    print("\nLoading hard samples from V9.2:")
    print(V92_HARD_JSON)

    with open(V92_HARD_JSON, "r") as f:
        hard_indices = json.load(f)

    print("Loaded hard samples:", len(hard_indices))

    # Copy record to V9.3 folder
    with open(f"{RUN_DIR}/hard_samples/train_hard_indices_used_from_v92.json", "w") as f:
        json.dump(hard_indices, f, indent=4)

    if os.path.exists(V92_HARD_CSV):
        try:
            df_tmp = pd.read_csv(V92_HARD_CSV)
            df_tmp.to_csv(f"{RUN_DIR}/hard_samples/train_hard_mining_results_used_from_v92.csv", index=False)
        except Exception as e:
            print("Could not copy V92 hard CSV:", e)

else:
    hard_indices, hard_df = mine_hard_samples(base_train_loader, split_name="train")

hard_set = set(hard_indices)

weights = np.ones(len(train_ds), dtype=np.float32) * NORMAL_SAMPLE_WEIGHT

for idx in hard_indices:
    if 0 <= idx < len(weights):
        weights[idx] = HARD_SAMPLE_WEIGHT

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

expected_hard_ratio = (len(hard_indices) * HARD_SAMPLE_WEIGHT) / (
    len(hard_indices) * HARD_SAMPLE_WEIGHT + (len(train_ds) - len(hard_indices)) * NORMAL_SAMPLE_WEIGHT
)

print("\nWeighted train loader created.")
print("Train batches:", len(train_loader))
print("Hard sample count:", len(hard_indices))
print("Hard sample raw ratio:", len(hard_indices) / max(1, len(train_ds)))
print("Expected weighted hard ratio:", expected_hard_ratio)
print("Hard sample weight:", HARD_SAMPLE_WEIGHT)
print("Normal sample weight:", NORMAL_SAMPLE_WEIGHT)

# ============================================================
# 9) TRAIN + EVAL
# ============================================================

def v93_score(m):
    penalty = 0.0

    if m["psnr_cover_y"] < MIN_COVER_Y:
        penalty += 80.0

    if m["psnr_secret_y"] < MIN_SECRET_Y:
        penalty += 90.0

    if m["ber_secret"] > MAX_BER:
        penalty += 90.0

    if m["avg_detection"] > MAX_AVG_DET:
        penalty += 70.0

    score = (
        0.65 * m["psnr_cover_y"] +
        3.30 * m["psnr_secret_y"] -
        145.0 * m["ber_secret"] +
        60.0 * m["avg_hidden"] -
        95.0 * m["old_xunet_stego_detection"] -
        55.0 * m["new_srnet_stego_detection"] -
        penalty
    )

    return score

def train_one_epoch(epoch, prev_secret_y=None, prev_ber=None):
    E.train()
    R.train()
    DX.train()
    DS.train()

    totals = {
        "E_loss":0, "R_loss":0, "DX_loss":0, "DS_loss":0,
        "DX_acc":0, "DS_acc":0,
        "cover_loss":0, "secret_loss":0,
        "edge":0, "fft":0, "residual":0, "color_stat":0,
        "mask_reg":0, "texture_mask":0,
        "adv_x":0, "adv_s":0, "mask_mean":0,
        "hard_batch_ratio":0
    }

    adv_scale = min(1.0, epoch / max(1, ADV_WARMUP_EPOCHS))

    if prev_secret_y is not None and prev_ber is not None:
        if prev_secret_y < SECRET_Y_GATE or prev_ber > BER_GATE:
            adv_scale *= 0.35

    lam_x = LAMBDA_OLD_XUNET_MAX * adv_scale
    lam_s = LAMBDA_NEW_SRNET_MAX * adv_scale

    pbar = tqdm(train_loader, desc=f"V9.3 Train epoch {epoch}/{EPOCHS}", leave=False)

    for cover, secret, idx in pbar:
        cover = cover.to(DEVICE, non_blocking=True)
        secret = secret.to(DEVICE, non_blocking=True)

        bs = cover.size(0)
        label_cover = torch.zeros(bs, 1, device=DEVICE)
        label_stego = torch.ones(bs, 1, device=DEVICE)

        hard_flags = torch.tensor(
            [1.0 if int(i.item()) in hard_set else 0.0 for i in idx],
            device=DEVICE
        ).view(bs, 1)

        hard_ratio = float(hard_flags.mean().item())

        # -------------------------
        # Train detectors
        # -------------------------
        with torch.no_grad():
            stego_det, _, _, _ = E(cover, secret)

        # XuNet: 1 step only in soft version
        set_detector_trainable(DX, True)
        opt_DX.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            lx_c = DX(cover)
            lx_s = DX(stego_det.detach())
            loss_DX = 0.5 * (bce(lx_c, label_cover) + bce(lx_s, label_stego))

        scaler_DX.scale(loss_DX).backward()
        scaler_DX.step(opt_DX)
        scaler_DX.update()

        with torch.no_grad():
            dx_acc = 0.5 * (
                (torch.sigmoid(lx_c) < 0.5).float().mean() +
                (torch.sigmoid(lx_s) >= 0.5).float().mean()
            )

        # SRNet: 1 step
        set_detector_trainable(DS, True)
        opt_DS.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            ls_c = DS(cover)
            ls_s = DS(stego_det.detach())
            loss_DS = 0.5 * (bce(ls_c, label_cover) + bce(ls_s, label_stego))

        scaler_DS.scale(loss_DS).backward()
        scaler_DS.step(opt_DS)
        scaler_DS.update()

        with torch.no_grad():
            ds_acc = 0.5 * (
                (torch.sigmoid(ls_c) < 0.5).float().mean() +
                (torch.sigmoid(ls_s) >= 0.5).float().mean()
            )

        # -------------------------
        # Train recovery network
        # -------------------------
        opt_R.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            stego_R, _, _, _ = E(cover, secret)
            recovered_R = R(stego_R.detach())
            R_loss, _, _, _ = secret_loss(secret, recovered_R)

        scaler_R.scale(R_loss).backward()
        scaler_R.unscale_(opt_R)
        torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
        scaler_R.step(opt_R)
        scaler_R.update()

        # -------------------------
        # Train embedding network
        # -------------------------
        set_detector_trainable(DX, False)
        set_detector_trainable(DS, False)

        opt_E.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            stego, residual, mask, alpha = E(cover, secret)
            recovered = R(stego)

            c_loss = F.mse_loss(stego, cover)
            s_loss, s_rgb, s_y, s_color = secret_loss(secret, recovered)

            e_loss = edge_loss(cover, stego)
            f_loss = fft_band_loss(cover, stego)
            r_loss = residual_distribution_loss(cover, stego)
            cs_loss = color_stat_loss(cover, stego)
            m_reg = mask_regularization(mask)
            tex_m = texture_guided_mask_loss(cover, mask)

            adv_x = bce(DX(stego), label_cover)
            adv_s = bce(DS(stego), label_cover)

            # softer hard multiplier
            hard_mult = 1.0 + 0.5 * hard_flags.mean()

            E_loss = (
                LAMBDA_COVER * c_loss +
                s_loss +
                LAMBDA_EDGE * e_loss +
                LAMBDA_FFT * f_loss +
                LAMBDA_RESIDUAL * r_loss +
                LAMBDA_COLOR_STAT * cs_loss +
                m_reg +
                0.20 * tex_m +
                hard_mult * (lam_x * adv_x + lam_s * adv_s)
            )

        scaler_E.scale(E_loss).backward()
        scaler_E.unscale_(opt_E)
        torch.nn.utils.clip_grad_norm_(E.parameters(), 1.0)
        scaler_E.step(opt_E)
        scaler_E.update()

        totals["E_loss"] += float(E_loss.item())
        totals["R_loss"] += float(R_loss.item())
        totals["DX_loss"] += float(loss_DX.item())
        totals["DS_loss"] += float(loss_DS.item())
        totals["DX_acc"] += float(dx_acc.item())
        totals["DS_acc"] += float(ds_acc.item())
        totals["cover_loss"] += float(c_loss.item())
        totals["secret_loss"] += float(s_loss.item())
        totals["edge"] += float(e_loss.item())
        totals["fft"] += float(f_loss.item())
        totals["residual"] += float(r_loss.item())
        totals["color_stat"] += float(cs_loss.item())
        totals["mask_reg"] += float(m_reg.item())
        totals["texture_mask"] += float(tex_m.item())
        totals["adv_x"] += float(adv_x.item())
        totals["adv_s"] += float(adv_s.item())
        totals["mask_mean"] += float(mask.mean().item())
        totals["hard_batch_ratio"] += hard_ratio

        pbar.set_postfix({
            "E": f"{E_loss.item():.4f}",
            "R": f"{R_loss.item():.4f}",
            "DX": f"{dx_acc.item():.2f}",
            "DS": f"{ds_acc.item():.2f}",
            "hard": f"{hard_ratio:.2f}",
            "a": f"{float(alpha.detach().cpu()):.4f}"
        })

    n = len(train_loader)

    for k in totals:
        totals[k] /= n

    totals["alpha"] = float(torch.clamp(E.hider.alpha.detach(), ALPHA_MIN, ALPHA_MAX).cpu())
    totals["lambda_old_xunet"] = lam_x
    totals["lambda_new_srnet"] = lam_s

    return totals

@torch.no_grad()
def evaluate(loader, split="val", save_panel=False, epoch=0):
    E.eval()
    R.eval()
    DX.eval()
    DS.eval()

    rows = []
    panel_saved = False

    pbar = tqdm(loader, desc=f"V9.3 Eval {split}", leave=False)

    for cover, secret, idx in pbar:
        cover = cover.to(DEVICE, non_blocking=True)
        secret = secret.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            stego, residual, mask, alpha = E(cover, secret)
            recovered = R(stego)

            cy, sy = rgb_to_y(cover), rgb_to_y(stego)
            sec_y, rec_y = rgb_to_y(secret), rgb_to_y(recovered)

            px_c = torch.sigmoid(DX(cover))
            px_s = torch.sigmoid(DX(stego))

            ps_c = torch.sigmoid(DS(cover))
            ps_s = torch.sigmoid(DS(stego))

            x_fp = (px_c >= 0.5).float().mean()
            x_det = (px_s >= 0.5).float().mean()
            x_hid = (px_s < 0.5).float().mean()

            s_fp = (ps_c >= 0.5).float().mean()
            s_det = (ps_s >= 0.5).float().mean()
            s_hid = (ps_s < 0.5).float().mean()

            m = {
                "cover_loss": F.mse_loss(stego, cover).item(),
                "secret_loss": F.mse_loss(recovered, secret).item(),

                "psnr_cover_y": batch_psnr(sy, cy).item(),
                "psnr_cover_rgb": batch_psnr(stego, cover).item(),
                "ssim_cover_y": simple_ssim(sy, cy).item(),
                "ncc_cover_y": ncc(sy, cy).item(),

                "psnr_secret_y": batch_psnr(rec_y, sec_y).item(),
                "psnr_secret_rgb": batch_psnr(recovered, secret).item(),
                "ssim_secret_y": simple_ssim(rec_y, sec_y).item(),
                "ncc_secret_y": ncc(rec_y, sec_y).item(),
                "ber_secret": ber_binary(secret, recovered).item(),

                "old_xunet_cover_false_positive": x_fp.item(),
                "old_xunet_stego_detection": x_det.item(),
                "old_xunet_stego_hidden": x_hid.item(),

                "new_srnet_cover_false_positive": s_fp.item(),
                "new_srnet_stego_detection": s_det.item(),
                "new_srnet_stego_hidden": s_hid.item(),

                "avg_detection": 0.5 * (x_det.item() + s_det.item()),
                "avg_hidden": 0.5 * (x_hid.item() + s_hid.item()),

                "mask_mean": mask.mean().item(),
                "alpha": float(alpha.detach().cpu())
            }

        rows.append(m)

        if save_panel and not panel_saved:
            path = f"{RUN_DIR}/best/best_epoch_{epoch}_{split}_V93_Cover_Secret_Stego_Recovered_Mask.png"
            save_titled_panel(cover, secret, stego, recovered, mask, path, epoch=epoch, split=split, n=4)
            panel_saved = True

    return pd.DataFrame(rows).mean().to_dict()

def print_epoch_report(epoch, train_m, val_m, score, row):
    print("\n" + "="*115)
    print(f"V9.3 EPOCH {epoch} RESULT")
    print("="*115)
    print(f"Embedding Loss        : {train_m['E_loss']:.6f}")
    print(f"Recovery Loss         : {train_m['R_loss']:.6f}")
    print(f"Old XuNet Train Acc   : {train_m['DX_acc']:.4f}")
    print(f"New SRNet Train Acc   : {train_m['DS_acc']:.4f}")
    print(f"Hard Batch Ratio      : {train_m['hard_batch_ratio']:.4f}")
    print("-"*115)
    print(f"Cover PSNR-Y          : {val_m['psnr_cover_y']:.4f}")
    print(f"Cover PSNR-RGB        : {val_m['psnr_cover_rgb']:.4f}")
    print(f"Cover SSIM-Y          : {val_m['ssim_cover_y']:.6f}")
    print(f"Cover NCC-Y           : {val_m['ncc_cover_y']:.6f}")
    print("-"*115)
    print(f"Secret PSNR-Y         : {val_m['psnr_secret_y']:.4f}")
    print(f"Secret PSNR-RGB       : {val_m['psnr_secret_rgb']:.4f}")
    print(f"Secret SSIM-Y         : {val_m['ssim_secret_y']:.6f}")
    print(f"Secret NCC-Y          : {val_m['ncc_secret_y']:.6f}")
    print(f"Secret BER            : {val_m['ber_secret']:.6f}")
    print("-"*115)
    print(f"Old XuNet Detection   : {val_m['old_xunet_stego_detection']:.4f}")
    print(f"Old XuNet Hidden      : {val_m['old_xunet_stego_hidden']:.4f}")
    print(f"New SRNet Detection   : {val_m['new_srnet_stego_detection']:.4f}")
    print(f"New SRNet Hidden      : {val_m['new_srnet_stego_hidden']:.4f}")
    print(f"Average Detection     : {val_m['avg_detection']:.4f}")
    print(f"Average Hidden        : {val_m['avg_hidden']:.4f}")
    print("-"*115)
    print(f"Mask Mean             : {val_m['mask_mean']:.6f}")
    print(f"Alpha                 : {val_m['alpha']:.6f}")
    print(f"V9.3 Score            : {score:.6f}")
    print(f"Time sec              : {row['time_sec']:.2f}")
    print("="*115 + "\n")

# ============================================================
# 10) MAIN LOOP
# ============================================================

history_path = f"{RUN_DIR}/metrics/training_log_v93.csv"

if os.path.exists(history_path):
    try:
        history = pd.read_csv(history_path).to_dict("records")
        print("Loaded previous history rows:", len(history))
    except:
        history = []
else:
    history = []

prev_secret_y = None
prev_ber = None

print("\n" + "="*100)
print("START V9.3 SOFT HARD-SAMPLE FINE-TUNING")
print("="*100)

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()

    train_m = train_one_epoch(epoch, prev_secret_y, prev_ber)
    val_m = evaluate(val_loader, split="val", save_panel=False, epoch=epoch)

    prev_secret_y = val_m["psnr_secret_y"]
    prev_ber = val_m["ber_secret"]

    sched_E.step()
    sched_R.step()
    sched_DX.step()
    sched_DS.step()

    score = v93_score(val_m)

    row = {
        "epoch": epoch,
        **{f"train_{k}": v for k, v in train_m.items()},
        **{f"val_{k}": v for k, v in val_m.items()},
        "v93_score": score,
        "lr_E": opt_E.param_groups[0]["lr"],
        "lr_R": opt_R.param_groups[0]["lr"],
        "lr_DX": opt_DX.param_groups[0]["lr"],
        "lr_DS": opt_DS.param_groups[0]["lr"],
        "time_sec": time.time() - t0
    }

    history.append(row)
    pd.DataFrame(history).to_csv(history_path, index=False)

    with open(f"{RUN_DIR}/metrics/latest_epoch_result_v93.json", "w") as f:
        json.dump(row, f, indent=4)

    improved = score > best_score

    if improved:
        best_score = score
        no_improve = 0

        _ = evaluate(val_loader, split="val", save_panel=True, epoch=epoch)

        torch.save({
            "epoch": epoch,
            "E": E.state_dict(),
            "R": R.state_dict(),
            "DX": DX.state_dict(),
            "DS": DS.state_dict(),
            "opt_E": opt_E.state_dict(),
            "opt_R": opt_R.state_dict(),
            "opt_DX": opt_DX.state_dict(),
            "opt_DS": opt_DS.state_dict(),
            "sched_E": sched_E.state_dict(),
            "sched_R": sched_R.state_dict(),
            "sched_DX": sched_DX.state_dict(),
            "sched_DS": sched_DS.state_dict(),
            "best_score": best_score,
            "no_improve": no_improve,
            "config": config,
            "val_metrics": val_m
        }, best_path)

        best_readable = {
            "best_epoch": int(epoch),
            "best_score": float(best_score),
            "cover_psnr_y": float(val_m["psnr_cover_y"]),
            "cover_psnr_rgb": float(val_m["psnr_cover_rgb"]),
            "cover_ssim_y": float(val_m["ssim_cover_y"]),
            "cover_ncc_y": float(val_m["ncc_cover_y"]),
            "secret_psnr_y": float(val_m["psnr_secret_y"]),
            "secret_psnr_rgb": float(val_m["psnr_secret_rgb"]),
            "secret_ssim_y": float(val_m["ssim_secret_y"]),
            "secret_ncc_y": float(val_m["ncc_secret_y"]),
            "secret_ber": float(val_m["ber_secret"]),
            "old_xunet_detection": float(val_m["old_xunet_stego_detection"]),
            "old_xunet_hidden": float(val_m["old_xunet_stego_hidden"]),
            "new_srnet_detection": float(val_m["new_srnet_stego_detection"]),
            "new_srnet_hidden": float(val_m["new_srnet_stego_hidden"]),
            "avg_detection": float(val_m["avg_detection"]),
            "avg_hidden": float(val_m["avg_hidden"]),
            "mask_mean": float(val_m["mask_mean"]),
            "alpha": float(val_m["alpha"])
        }

        pd.DataFrame([best_readable]).to_csv(f"{RUN_DIR}/best/best_readable_summary_v93.csv", index=False)

        with open(f"{RUN_DIR}/best/best_readable_summary_v93.json", "w") as f:
            json.dump(best_readable, f, indent=4)

        print(">>> NEW V9.3 BEST SAVED")

    else:
        no_improve += 1

    torch.save({
        "epoch": epoch,
        "E": E.state_dict(),
        "R": R.state_dict(),
        "DX": DX.state_dict(),
        "DS": DS.state_dict(),
        "opt_E": opt_E.state_dict(),
        "opt_R": opt_R.state_dict(),
        "opt_DX": opt_DX.state_dict(),
        "opt_DS": opt_DS.state_dict(),
        "sched_E": sched_E.state_dict(),
        "sched_R": sched_R.state_dict(),
        "sched_DX": sched_DX.state_dict(),
        "sched_DS": sched_DS.state_dict(),
        "best_score": best_score,
        "no_improve": no_improve,
        "config": config
    }, last_path)

    if epoch % SAVE_EVERY == 0:
        torch.save({
            "epoch": epoch,
            "E": E.state_dict(),
            "R": R.state_dict(),
            "DX": DX.state_dict(),
            "DS": DS.state_dict(),
            "config": config
        }, f"{RUN_DIR}/checkpoints/epoch_{epoch}_v93.pt")

    print(
        f"Epoch [{epoch}/{EPOCHS}] | "
        f"coverY={val_m['psnr_cover_y']:.2f} | "
        f"secretY={val_m['psnr_secret_y']:.2f} | "
        f"secretRGB={val_m['psnr_secret_rgb']:.2f} | "
        f"BER={val_m['ber_secret']:.4f} | "
        f"OldXuDet={val_m['old_xunet_stego_detection']:.4f} | "
        f"NewSRDet={val_m['new_srnet_stego_detection']:.4f} | "
        f"AvgDet={val_m['avg_detection']:.4f} | "
        f"AvgHidden={val_m['avg_hidden']:.4f} | "
        f"HardRatio={train_m['hard_batch_ratio']:.4f} | "
        f"Mask={val_m['mask_mean']:.4f} | "
        f"alpha={val_m['alpha']:.5f} | "
        f"score={score:.4f} | "
        f"best={best_score:.4f} | "
        f"no_improve={no_improve}/{PATIENCE} | "
        f"time={row['time_sec']:.1f}s"
    )

    print_epoch_report(epoch, train_m, val_m, score, row)

    if no_improve >= PATIENCE:
        print(f"\nEARLY STOPPING: no improvement for {PATIENCE} epochs.")
        break

print("\nV9.3 fine-tuning finished.")
print("Best checkpoint:", best_path)
print("Last checkpoint :", last_path)

# ============================================================
# 11) FINAL TEST
# ============================================================

print("\n" + "="*100)
print("LOADING BEST V9.3 CHECKPOINT FOR FINAL TEST")
print("="*100)

ckpt = torch.load(best_path, map_location=DEVICE)

E.load_state_dict(ckpt["E"])
R.load_state_dict(ckpt["R"])
DX.load_state_dict(ckpt["DX"])
DS.load_state_dict(ckpt["DS"])

best_epoch = ckpt["epoch"]
best_val_score = ckpt["best_score"]

print("Best V9.3 epoch:", best_epoch)
print("Best V9.3 val score:", best_val_score)

test_m = evaluate(test_loader, split="test", save_panel=True, epoch=best_epoch)
test_score = v93_score(test_m)

final_summary = {
    "version": "V9.3 Soft Hard-Sample Fine-Tuning",
    "best_epoch": int(best_epoch),
    "best_val_score": float(best_val_score),
    "test_score": float(test_score),

    "cover_psnr_y": float(test_m["psnr_cover_y"]),
    "cover_psnr_rgb": float(test_m["psnr_cover_rgb"]),
    "cover_ssim_y": float(test_m["ssim_cover_y"]),
    "cover_ncc_y": float(test_m["ncc_cover_y"]),

    "secret_psnr_y": float(test_m["psnr_secret_y"]),
    "secret_psnr_rgb": float(test_m["psnr_secret_rgb"]),
    "secret_ssim_y": float(test_m["ssim_secret_y"]),
    "secret_ncc_y": float(test_m["ncc_secret_y"]),
    "secret_ber": float(test_m["ber_secret"]),

    "old_xunet_detection": float(test_m["old_xunet_stego_detection"]),
    "old_xunet_hidden": float(test_m["old_xunet_stego_hidden"]),
    "new_srnet_detection": float(test_m["new_srnet_stego_detection"]),
    "new_srnet_hidden": float(test_m["new_srnet_stego_hidden"]),
    "avg_detection": float(test_m["avg_detection"]),
    "avg_hidden": float(test_m["avg_hidden"]),

    "mask_mean": float(test_m["mask_mean"]),
    "alpha": float(test_m["alpha"]),
    "num_hard_train_samples": int(len(hard_indices)),
    "hard_train_ratio": float(len(hard_indices) / max(1, len(train_ds))),
    "hard_sample_weight": float(HARD_SAMPLE_WEIGHT),
    "expected_weighted_hard_ratio": float(expected_hard_ratio)
}

with open(f"{RUN_DIR}/metrics/final_test_summary_v93.json", "w") as f:
    json.dump(final_summary, f, indent=4)

pd.DataFrame([final_summary]).to_csv(f"{RUN_DIR}/metrics/final_test_summary_v93.csv", index=False)

with open(f"{RUN_DIR}/metrics/final_test_summary_v93.txt", "w") as f:
    for k, v in final_summary.items():
        f.write(f"{k}: {v}\n")

print("\n" + "="*100)
print("FINAL TEST SUMMARY - V9.3")
print("="*100)

for k, v in final_summary.items():
    if isinstance(v, float):
        print(f"{k}: {v:.6f}")
    else:
        print(f"{k}: {v}")

print("\nSaved files:")
print("RUN_DIR:", RUN_DIR)
print("Best checkpoint:", best_path)
print("Last checkpoint:", last_path)
print("Training log:", f"{RUN_DIR}/metrics/training_log_v93.csv")
print("Hard indices used:", f"{RUN_DIR}/hard_samples/train_hard_indices_used_from_v92.json")
print("Best summary CSV:", f"{RUN_DIR}/best/best_readable_summary_v93.csv")
print("Final summary CSV:", f"{RUN_DIR}/metrics/final_test_summary_v93.csv")
print("Final summary JSON:", f"{RUN_DIR}/metrics/final_test_summary_v93.json")
print("Final summary TXT:", f"{RUN_DIR}/metrics/final_test_summary_v93.txt")
print("Panels saved in:", f"{RUN_DIR}/best")