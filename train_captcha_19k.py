# train_captcha_19k.py
"""
Training script for 19k+ CAPTCHA dataset
Usage:
  python train_captcha_19k.py
"""

import os
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# Install albumentations if not present:
# pip install albumentations==1.3.0
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For confusion matrix & save
from sklearn.metrics import confusion_matrix

# ========== CONFIG ==========
# Use relative paths for portability
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "19k plus captchas")  # 19k+ captchas directory
TRAIN_LIST = os.path.join(CURRENT_DIR, "data_19k", "train_labels.txt")  # 19k dataset train labels
VAL_LIST = os.path.join(CURRENT_DIR, "data_19k", "val_labels.txt")  # 19k dataset val labels
OUT_DIR = "runs/exp_19k"  # New experiment directory for 19k dataset
IMAGE_SIZE = (128, 64)  # width, height
BATCH_SIZE = 64  # Increased batch size for larger dataset
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues with corrupted images
NUM_EPOCHS = 200  # Reasonable epochs for larger dataset
LR = 3e-4  # Slightly lower learning rate for stability
WEIGHT_DECAY = 1e-4
CLIP_NORM = 0.5
USE_AMP = True
SAVE_EVERY = 10  # Save every 10 epochs
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NUM_HEADS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKGROUND_TEMPLATE = None
SEED = 42
# Label smoothing and focal loss parameters
LABEL_SMOOTHING = 0.1
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
USE_FOCAL_LOSS = True
# Position-weighted loss parameters
POSITION_WEIGHTS = [1.5, 1.0, 1.0, 1.0, 1.0, 1.2]  # Boost first and last positions
# ============================

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Derived
VOCAB = list(CHARSET)
CLS = len(VOCAB)  # should be 62

char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for c, i in char2idx.items()}

# Filter out characters that don't exist in the dataset
def get_valid_charset(train_list):
    """Get only characters that actually exist in the training data"""
    all_chars = set()
    for _, label in train_list:
        all_chars.update(label)
    
    # Keep only characters that exist in both CHARSET and training data
    valid_chars = sorted([c for c in CHARSET if c in all_chars])
    print(f"Valid characters in dataset: {len(valid_chars)} out of {len(CHARSET)}")
    print(f"Missing characters: {set(CHARSET) - all_chars}")
    return valid_chars


# ========== Dataset ==========
def load_label_list(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            if "\t" in l:
                img, lbl = l.split("\t")
            elif "," in l:
                img, lbl = l.split(",")
            else:
                parts = l.split()
                img, lbl = parts[0], parts[1]
            lbl = lbl.strip()
            lines.append((img.strip(), lbl))
    return lines

class CaptchaDataset(Dataset):
    def __init__(self, label_list, image_size=(128,64), transform=None, background_template=None, char_mapping=None):
        self.items = label_list
        self.transform = transform
        self.image_size = image_size
        self.char_mapping = char_mapping or char2idx
        self.bg_template = None
        if background_template:
            bg = Image.open(background_template).convert("L").resize(image_size)
            self.bg_template = np.array(bg).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        
        # Handle corrupted images gracefully
        try:
            img = Image.open(img_path).convert("L")  # grayscale
        except Exception as e:
            print(f"⚠️ Skipping corrupted image: {img_path} - {e}")
            # Return a random valid image instead
            valid_idx = (idx + 1) % len(self.items)
            return self.__getitem__(valid_idx)
        
        img = img.resize(self.image_size)
        arr = np.array(img).astype(np.float32) / 255.0  # 0..1

        # optional background subtraction (simple)
        if self.bg_template is not None:
            arr = arr - self.bg_template
            arr = np.clip(arr, 0.0, 1.0)

        # to uint8 for albumentations
        arr8 = (arr * 255).astype(np.uint8)
        if self.transform:
            augmented = self.transform(image=arr8)
            img_t = augmented["image"]  # tensor CHW
        else:
            # convert to tensor
            img_t = transforms.ToTensor()(Image.fromarray(arr8))

        # label to indices - IMPORTANT: Filter out invalid characters
        if len(label) != NUM_HEADS:
            # pad or trim if necessary (but ideally labels are exact)
            label = label[:NUM_HEADS].ljust(NUM_HEADS, " ")
        
        # Map characters to new indices, skip characters not in valid set
        targets = []
        for ch in label:
            if ch in self.char_mapping:
                targets.append(self.char_mapping[ch])
            else:
                # Replace invalid characters with a space or most common character
                targets.append(self.char_mapping.get(' ', 0))  # fallback to index 0
        
        targets = torch.LongTensor(targets)

        return img_t, targets, img_path, label


# ========== Augmentations ==========
def get_transforms(phase="train"):
    w, h = IMAGE_SIZE
    if phase == "train":
        return A.Compose([
            A.Resize(height=h, width=w),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.GaussianBlur(p=0.3),
                A.GaussNoise(p=0.3),
            ], p=0.6),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.03, rotate_limit=6, p=0.6),  # Increased shift limit
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.OneOf([A.CLAHE(), A.Equalize()], p=0.3),
            A.CoarseDropout(p=0.4),  # Increased occlusion
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=h, width=w),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])


# ========== Model ==========
class MultiHeadResNet(nn.Module):
    def __init__(self, num_heads=6, num_classes=62, pretrained=False):
        super().__init__()
        # Use ResNet18 and adapt to 1-channel input
        # Updated to use weights enum as suggested by ChatGPT
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
        except ImportError:
            # Fallback for older torchvision versions
            self.backbone = models.resnet18(pretrained=pretrained)
        
        # change first conv to accept 1 channel
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # remove final fc
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        # heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(nn.Sequential(
                nn.Linear(in_feat, 256),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(256, num_classes)
            ))

    def forward(self, x):
        # x: [B,1,H,W]
        feat = self.backbone(x)  # [B, in_feat]
        outs = []
        for h in self.heads:
            outs.append(h(feat))  # [B, num_classes]
        # return list of heads or stacked tensor
        out = torch.stack(outs, dim=1)  # [B, num_heads, num_classes]
        return out


# ========== Losses & Metrics ==========
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_position_weighted_loss(logits, targets, position_weights=None, use_focal=False, label_smoothing=0.0, class_weights=None):
    """
    Compute position-weighted loss for multi-head classification
    logits: [B, num_heads, num_classes]
    targets: [B, num_heads] (Long)
    position_weights: [num_heads] - weight for each position
    """
    if position_weights is None:
        position_weights = torch.tensor(POSITION_WEIGHTS, device=logits.device)
    
    B, H, C = logits.shape
    logits = logits.view(B * H, C)
    targets_flat = targets.view(B * H)
    
    # Compute loss per sample
    if use_focal:
        loss_fn = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='none')
        losses = loss_fn(logits, targets_flat)
    else:
        if label_smoothing > 0.0:
            losses = F.cross_entropy(logits, targets_flat, weight=class_weights, label_smoothing=label_smoothing, reduction='none')
        else:
            losses = F.cross_entropy(logits, targets_flat, weight=class_weights, reduction='none')
    
    # Reshape losses to [B, H]
    losses = losses.view(B, H)
    
    # Apply position weights
    position_weights = position_weights.unsqueeze(0).expand(B, H)  # [B, H]
    weighted_losses = losses * position_weights
    
    # Return mean of weighted losses
    return weighted_losses.mean()

def compute_loss_and_metrics(logits, targets, ignore_index=None, class_weights=None, use_focal=False, label_smoothing=0.0):
    """
    logits: [B, num_heads, num_classes]
    targets: [B, num_heads] (Long)
    """
    B, H, C = logits.shape
    device = logits.device
    logits = logits.view(B * H, C)
    targets_flat = targets.view(B * H)
    
    if use_focal:
        # Use focal loss
        loss_fn = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        loss = loss_fn(logits, targets_flat)
    else:
        # Use cross entropy with label smoothing
        if label_smoothing > 0.0:
            loss = F.cross_entropy(logits, targets_flat, weight=class_weights, label_smoothing=label_smoothing)
        else:
            loss = F.cross_entropy(logits, targets_flat, weight=class_weights)

    preds = logits.argmax(dim=-1).view(B, H)
    correct = (preds == targets).float()
    per_head_acc = correct.mean(dim=0).cpu().numpy().tolist()
    # full-string accuracy
    full_acc = (correct.prod(dim=1)).mean().item()
    return loss, preds, per_head_acc, full_acc


# ========== Utilities ==========
def idxs_to_string(idxs, idx_to_char_map=None):
    if idx_to_char_map is None:
        idx_to_char_map = idx2char
    return "".join(idx_to_char_map.get(int(i), '?') for i in idxs)


def save_checkpoint(state, path):
    torch.save(state, path)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, scheduler=None, class_weights=None):
    model.train()
    running_loss = 0.0
    running_full_acc = 0.0
    running_per_head = np.zeros(NUM_HEADS)
    n = 0
    for batch_idx, (imgs, targets, *_ ) in enumerate(loader):
        imgs = imgs.to(device)  # [B,1,H,W]
        targets = targets.to(device)  # [B, H]
        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=USE_AMP):
            logits = model(imgs)  # [B,H,C]
            # Use position-weighted loss as suggested by ChatGPT
            loss = compute_position_weighted_loss(
                logits, targets, position_weights=None,  # Will use POSITION_WEIGHTS
                use_focal=USE_FOCAL_LOSS, label_smoothing=LABEL_SMOOTHING,
                class_weights=class_weights
            )
            # For metrics, still use standard loss
            _, preds, per_head_acc, full_acc = compute_loss_and_metrics(
                logits, targets, class_weights=class_weights, 
                use_focal=USE_FOCAL_LOSS, label_smoothing=LABEL_SMOOTHING
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        running_full_acc += full_acc * imgs.size(0)
        running_per_head += np.array(per_head_acc) * imgs.size(0)
        n += imgs.size(0)

        if batch_idx % 100 == 0:  # Print every 100 batches for larger dataset
            print(f"Epoch {epoch} Batch {batch_idx}/{len(loader)} loss={loss.item():.4f} full_acc={full_acc:.4f}")

    if scheduler is not None:
        scheduler.step(running_loss / max(1, n))

    avg_loss = running_loss / max(1, n)
    avg_full = running_full_acc / max(1, n)
    avg_per_head = (running_per_head / max(1, n)).tolist()
    return avg_loss, avg_per_head, avg_full


def validate(model, loader, device, topk_mispreds=50, idx_to_char_map=None):
    model.eval()
    running_loss = 0.0
    running_full_acc = 0.0
    running_per_head = np.zeros(NUM_HEADS)
    n = 0
    all_preds = []
    all_trues = []
    mispreds = []
    with torch.no_grad():
        for imgs, targets, paths, labels in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            loss, preds, per_head_acc, full_acc = compute_loss_and_metrics(
                logits, targets, use_focal=USE_FOCAL_LOSS, label_smoothing=LABEL_SMOOTHING
            )
            running_loss += loss.item() * imgs.size(0)
            running_full_acc += full_acc * imgs.size(0)
            running_per_head += np.array(per_head_acc) * imgs.size(0)
            n += imgs.size(0)
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            all_preds.append(preds_np)
            all_trues.append(targets_np)
            # collect some mispredictions
            for i in range(preds_np.shape[0]):
                if not np.array_equal(preds_np[i], targets_np[i]) and len(mispreds) < topk_mispreds:
                    mispreds.append((paths[i], labels[i], idxs_to_string(preds_np[i], idx_to_char_map)))

    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    avg_loss = running_loss / max(1, n)
    avg_full = running_full_acc / max(1, n)
    avg_per_head = (running_per_head / max(1, n)).tolist()

    # compute confusion matrices (per head)
    confs = []
    for h in range(NUM_HEADS):
        try:
            cm = confusion_matrix(all_trues[:, h], all_preds[:, h], labels=list(range(model.heads[0][-1].out_features)))
        except Exception as e:
            cm = None
        confs.append(cm)

    return avg_loss, avg_per_head, avg_full, confs, mispreds


# ========== MAIN ==========
def main():
    print("🚀 Starting 19k+ CAPTCHA Training")
    print("=" * 60)
    print("Device:", DEVICE)
    
    # load lists
    train_list = load_label_list(TRAIN_LIST)
    val_list = load_label_list(VAL_LIST)
    print(f"📊 Train samples: {len(train_list)}, Val samples: {len(val_list)}")
    
    # Get valid characters and create mappings FIRST
    valid_chars = get_valid_charset(train_list)
    char2idx_new = {c: i for i, c in enumerate(valid_chars)}
    idx2char_new = {i: c for c, i in char2idx_new.items()}
    CLS_NEW = len(valid_chars)
    
    print(f"🔤 Updated character set: {CLS_NEW} characters")
    print(f"Characters: {''.join(valid_chars)}")

    # build datasets with updated character mapping
    train_ds = CaptchaDataset(train_list, image_size=IMAGE_SIZE, transform=get_transforms("train"),
                              background_template=BACKGROUND_TEMPLATE, char_mapping=char2idx_new)
    val_ds = CaptchaDataset(val_list, image_size=IMAGE_SIZE, transform=get_transforms("val"),
                            background_template=BACKGROUND_TEMPLATE, char_mapping=char2idx_new)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # show class distribution
    all_targets = [t for _, t in train_list]
    cnts = Counter("".join(all_targets))
    print("📈 Top charset counts (train):")
    for c, num in cnts.most_common(10):
        print(f"  {c}: {num}")

    # class weights for imbalanced classes (only for characters that exist)
    total_chars = sum(cnts.values())
    class_weights = []
    for c in valid_chars:
        count = cnts.get(c, 1)  # Should always exist now
        weight = total_chars / (count * len(valid_chars))  # Normalize by number of classes
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    # Check for extreme weights and cap them
    max_weight = class_weights.max().item()
    min_weight = class_weights.min().item()
    weight_ratio = max_weight / min_weight
    
    print(f"⚖️ Class weight ratio: {weight_ratio:.2f}")
    if weight_ratio > 10.0:  # Cap extreme weights
        median_weight = class_weights.median()
        cap_value = median_weight * 5.0
        class_weights = torch.clamp(class_weights, max=cap_value)
        print(f"🔒 Capped extreme weights at {cap_value:.2f}")
    
    print(f"✅ Using balanced class weights for training.")
    print(f"🔤 Valid characters: {len(valid_chars)}")

    # model with updated number of classes
    model = MultiHeadResNet(num_heads=NUM_HEADS, num_classes=CLS_NEW, pretrained=False).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-8)

    # Improved learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=LR/10
    )

    # Load checkpoint if exists to resume training
    start_epoch = 1
    last_checkpoint = os.path.join(OUT_DIR, "last.pth")
    if os.path.exists(last_checkpoint):
        checkpoint = torch.load(last_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("best_val", 0.0)
        print(f"🔄 Resumed from epoch {checkpoint['epoch']}, best_val: {best_val}")
    else:
        best_val = 0.0
    scaler = GradScaler(enabled=USE_AMP)
    
    print(f"\n🎯 Training Configuration:")
    print(f"  📊 Dataset: 19k+ CAPTCHAs ({len(train_list)} train, {len(val_list)} val)")
    print(f"  🔄 Epochs: {NUM_EPOCHS}")
    print(f"  📈 Learning rate: {LR}")
    print(f"  📦 Batch size: {BATCH_SIZE}")
    print(f"  🎯 Use focal loss: {USE_FOCAL_LOSS}")
    print(f"  🏷️ Label smoothing: {LABEL_SMOOTHING}")
    print(f"  🔤 Valid characters: {len(valid_chars)}")
    print(f"  💾 Output directory: {OUT_DIR}")
    print(f"  📍 Position weights: {POSITION_WEIGHTS}")
    print("=" * 60)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss, train_per_head, train_full = train_one_epoch(
            model, train_loader, optimizer, scaler, DEVICE, epoch, 
            scheduler=None, class_weights=class_weights
        )
        val_loss, val_per_head, val_full, confs, mispreds = validate(
            model, val_loader, DEVICE, topk_mispreds=200, idx_to_char_map=idx2char_new
        )

        print(f"📊 Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_full:.4f} | Per-Head: {[f'{acc:.3f}' for acc in train_per_head]}")
        print(f"📊 Epoch {epoch:3d} | Val Loss:   {val_loss:.4f} | Val Acc:   {val_full:.4f} | Per-Head:   {[f'{acc:.3f}' for acc in val_per_head]}")
        print(f"📈 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        # save checkpoint if best full-string accuracy
        if val_full > best_val:
            best_val = val_full
            path = os.path.join(OUT_DIR, f"best_epoch_{epoch:03d}_fullacc_{val_full:.4f}.pth")
            try:
                save_checkpoint({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": best_val,
                }, path)
                print(f"💾 Saved best model to {path}")
            except Exception as e:
                print(f"❌ Failed to save best model: {e}")

        # save last
        if epoch % SAVE_EVERY == 0:
            last_path = os.path.join(OUT_DIR, "last.pth")
            try:
                save_checkpoint({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": best_val,
                    "char_mapping": {"char2idx": char2idx_new, "idx2char": idx2char_new},
                }, last_path)
                print(f"💾 Saved checkpoint to {last_path}")
            except Exception as e:
                print(f"❌ Failed to save checkpoint: {e}")

        # write mispreds to disk
        mispred_file = os.path.join(OUT_DIR, f"mispreds_epoch_{epoch:03d}.txt")
        with open(mispred_file, "w", encoding="utf-8") as f:
            for p, true_lbl, pred in mispreds:
                f.write(f"{p}\t{true_lbl}\t{pred}\n")
        print(f"📝 Saved {len(mispreds)} mispredictions to {mispred_file}")

    print("🎉 Training finished!")
    print(f"🏆 Best validation accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()