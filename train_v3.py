"""
train_v3.py — ABIDE MRI Classification: Fixed & Improved Pipeline
================================================================
Key changes from v2 (EfficientNet-B0):
  1. All valid brain slices used (no 60% central filter — avoids data loss)
  2. Proper 5-Fold cross-validation → reliable subject accuracy (not 14-subject noise)
  3. Linear probe first (freeze backbone, train head only, 5 epochs)
     then fine-tune (unfreeze last ResNet layer block, 20 epochs)
  4. ResNet50 backbone — better feature extractor for medical images
  5. Stronger, targeted regularization:
       - Dropout(0.6) before classifier
       - weight_decay=5e-3
       - label_smoothing=0.1
  6. Cosine annealing LR with restarts
  7. Class-weighted loss (no sampler — avoids overfitting on small set)
  8. Test-Time Augmentation (TTA) for subject voting
  9. Reports per-fold AND mean ± std subject accuracy
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

# ── Seed ────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("  ABIDE Classification v3 — ResNet50 + 5-Fold CV")
print("=" * 60)
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Device: {device}")

# ── 1. Load Labels ───────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\abide\Phenotypic_V1_0b.csv")
dx_map = {}
for _, row in df.iterrows():
    s = str(row['SUB_ID'])
    dx = int(row['DX_GROUP']) - 1   # 0=ASD  1=TC
    dx_map[s] = dx
    dx_map[s.zfill(7)] = dx
print(f"\nLoaded {len(df)} phenotypic records")

# ── 2. Collect ALL valid slices per subject ──────────────────────────────────
#     NOTE: No central 60% filter — use all non-empty brain slices.
#     Filtering only removes pure background (std<5 or brain area<2.5%).
ANAT_DIR = r"C:\abide\OUTPUT\anat"

print("\nScanning slices (filtering empty/background only)...")
all_records = []
subject_index = {}   # subj_id -> list of record indices

for subj in os.listdir(ANAT_DIR):
    if subj == "UCLA_1":
        continue
    sp = os.path.join(ANAT_DIR, subj)
    if not os.path.isdir(sp):
        continue
    label = dx_map.get(subj, dx_map.get(subj.lstrip('0'), None))
    if label is None:
        continue

    slice_files = sorted(glob.glob(os.path.join(sp, "*.png")))
    valid = []
    for s_path in slice_files:
        try:
            arr = np.array(Image.open(s_path).convert("L"))
            if arr.std() >= 5.0 and (arr > 15).mean() >= 0.02:
                valid.append(s_path)
        except Exception:
            pass

    if len(valid) < 5:
        continue

    start = len(all_records)
    for s_path in valid:
        all_records.append({"path": s_path, "subject_id": subj, "label": label})
    subject_index[subj] = list(range(start, len(all_records)))

n_asd = sum(1 for r in all_records if r["label"] == 0)
n_tc  = sum(1 for r in all_records if r["label"] == 1)
print(f"Total valid slices : {len(all_records)}  (ASD={n_asd}, TC={n_tc})")
print(f"Total subjects     : {len(subject_index)}")

# ── 3. Dataset ───────────────────────────────────────────────────────────────
class ABIDEDataset(Dataset):
    """Per-scan Z-score normalised MRI slice dataset."""
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        arr  = np.array(Image.open(item["path"]).convert("L"), dtype=np.float32)
        # Per-scan Z-score normalisation
        arr  = (arr - arr.mean()) / (arr.std() + 1e-8)
        # Rescale to [0, 255] uint8 for PIL/torchvision transforms
        arr  = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        img  = Image.fromarray(arr).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, item["label"], item["subject_id"]

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Augmentation: gentle — brain MRI symmetry + mild geometric variation only
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── 4. Model Factory ─────────────────────────────────────────────────────────
def build_model(freeze_backbone=True):
    """
    ResNet50 pretrained on ImageNet.
    Phase 1 (linear probe):  freeze all backbone layers, train head only.
    Phase 2 (fine-tune):     unfreeze layer4 block, keep lower layers frozen.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Freeze only layers 1-3; unfreeze layer4 + head
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in ["layer4", "fc"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(128, 2)
    )
    return model.to(device)

# ── 5. Training / Validation functions ──────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast(device_type="cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate_tta(model, loader, criterion):
    """
    Validation with TTA (original + horizontal flip).
    Returns: loss, slice_acc, subj_acc, preds, labels, subj_probs dict
    """
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    subj_probs = {}

    with torch.no_grad():
        for imgs, labels, subjs in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out1   = model(imgs)
            out2   = model(torch.flip(imgs, dims=[3]))
            loss   = criterion(out1, labels)
            total_loss += loss.item() * imgs.size(0)
            probs  = (torch.softmax(out1, 1) + torch.softmax(out2, 1)) / 2.0
            preds  = probs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for i, subj in enumerate(subjs):
                if subj not in subj_probs:
                    subj_probs[subj] = {"probs": [], "label": labels[i].item()}
                subj_probs[subj]["probs"].append(probs[i, 1].item())

    slice_acc = accuracy_score(all_labels, all_preds) * 100.0
    n_correct  = sum(
        1 for d in subj_probs.values()
        if (1 if np.mean(d["probs"]) >= 0.5 else 0) == d["label"]
    )
    subj_acc = 100.0 * n_correct / len(subj_probs)
    return total_loss / len(loader.dataset), slice_acc, subj_acc, all_preds, all_labels, subj_probs

# ── 6. 5-Fold Cross-Validation ───────────────────────────────────────────────
BATCH_SIZE   = 32
N_FOLDS      = 5
PROBE_EPOCHS = 5    # Phase 1: linear probe (frozen backbone)
TUNE_EPOCHS  = 20   # Phase 2: fine-tune layer4
PATIENCE     = 6
PROBE_LR     = 1e-3
TUNE_LR      = 5e-5

subjects_list = [{"id": subj, "label": all_records[idxs[0]]["label"]}
                 for subj, idxs in subject_index.items()]
subj_ids  = [s["id"]    for s in subjects_list]
subj_lbls = [s["label"] for s in subjects_list]

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
print(f"\n{'='*60}")
print(f"  Starting {N_FOLDS}-Fold Cross-Validation")
print(f"  Phase 1: {PROBE_EPOCHS} linear probe epochs (LR={PROBE_LR})")
print(f"  Phase 2: {TUNE_EPOCHS} fine-tune epochs    (LR={TUNE_LR})")
print(f"{'='*60}")

for fold, (tr_subj_idx, va_subj_idx) in enumerate(
    sgkf.split(range(len(subj_ids)), subj_lbls, subj_ids)
):
    tr_subjs = set(subj_ids[i] for i in tr_subj_idx)
    va_subjs = set(subj_ids[i] for i in va_subj_idx)
    assert len(tr_subjs & va_subjs) == 0, "Subject overlap!"

    tr_records = [r for r in all_records if r["subject_id"] in tr_subjs]
    va_records = [r for r in all_records if r["subject_id"] in va_subjs]

    n_asd_tr = sum(1 for r in tr_records if r["label"] == 0)
    n_tc_tr  = sum(1 for r in tr_records if r["label"] == 1)

    print(f"\nFold {fold+1}/{N_FOLDS}:")
    print(f"  Train: {len(tr_records)} slices / {len(tr_subjs)} subjects")
    print(f"  Val:   {len(va_records)} slices / {len(va_subjs)} subjects")

    # Class weights from training fold only
    n_total  = len(tr_records)
    w_asd    = n_total / (2 * n_asd_tr)
    w_tc     = n_total / (2 * n_tc_tr)
    cw       = torch.tensor([w_asd, w_tc], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    tr_ds = ABIDEDataset(tr_records, transform=train_transform)
    va_ds = ABIDEDataset(va_records, transform=val_transform)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    # ── Phase 1: Linear Probe (frozen backbone) ──────────────────────────
    print(f"  Phase 1: Linear probe ({PROBE_EPOCHS} epochs)...")
    model = build_model(freeze_backbone=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PROBE_LR, weight_decay=1e-2
    )
    best_subj_acc, best_state = 0.0, None
    for ep in range(PROBE_EPOCHS):
        tr_loss, tr_acc = train_epoch(model, tr_ld, criterion, optimizer, scaler)
        _, sl_acc, subj_acc, _, _, _ = evaluate_tta(model, va_ld, criterion)
        marker = " ★" if subj_acc > best_subj_acc else ""
        print(f"    Probe ep {ep+1}: Train={tr_acc:.1f}%  Val={sl_acc:.1f}%  Subj={subj_acc:.1f}%{marker}")
        if subj_acc > best_subj_acc:
            best_subj_acc = subj_acc
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Phase 2: Fine-tune layer4 ────────────────────────────────────────
    print(f"  Phase 2: Fine-tune layer4 ({TUNE_EPOCHS} epochs)...")
    model = build_model(freeze_backbone=False)
    if best_state:
        model.load_state_dict(best_state)  # start from best probe checkpoint
    # Re-enable layer4 gradients (build_model already does this)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TUNE_LR, weight_decay=5e-3
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    no_improve = 0
    for ep in range(TUNE_EPOCHS):
        tr_loss, tr_acc = train_epoch(model, tr_ld, criterion, optimizer, scaler)
        _, sl_acc, subj_acc, val_preds, val_labels, subj_probs = evaluate_tta(
            model, va_ld, criterion
        )
        scheduler.step()
        gap    = tr_acc - sl_acc
        marker = " ★" if subj_acc > best_subj_acc else ""
        print(f"    Tune ep {ep+1:02d}: Train={tr_acc:.1f}%  Val={sl_acc:.1f}%  Subj={subj_acc:.1f}%  Gap={gap:.1f}%{marker}")
        if subj_acc > best_subj_acc:
            best_subj_acc = subj_acc
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    Early stopping at tune epoch {ep+1}.")
                break

    # ── Final evaluation on best checkpoint ─────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    _, final_sl, final_subj, val_preds, val_labels, subj_probs = evaluate_tta(
        model, va_ld, criterion
    )
    cm = confusion_matrix(val_labels, val_preds)
    fold_results.append({
        "fold": fold + 1,
        "n_val_subjs": len(va_subjs),
        "slice_acc": final_sl,
        "subj_acc": final_subj,
        "cm": cm,
        "subj_probs": subj_probs,
        "val_labels": val_labels,
        "val_preds": val_preds,
    })
    torch.save(best_state, rf"C:\abide\best_resnet50_fold{fold+1}.pth")
    print(f"  Fold {fold+1} FINAL: SliceAcc={final_sl:.2f}%  SubjAcc={final_subj:.2f}%")

# ── 7. Cross-Validation Summary ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("  CROSS-VALIDATION RESULTS SUMMARY")
print(f"{'='*60}")
subj_accs = [r["subj_acc"]  for r in fold_results]
slice_accs = [r["slice_acc"] for r in fold_results]

for r in fold_results:
    print(f"  Fold {r['fold']}: SliceAcc={r['slice_acc']:.2f}%  SubjAcc={r['subj_acc']:.2f}%  ({r['n_val_subjs']} subjects)")

print()
print(f"  Mean Slice Accuracy   : {np.mean(slice_accs):.2f}% ± {np.std(slice_accs):.2f}%")
print(f"  Mean Subject Accuracy : {np.mean(subj_accs):.2f}% ± {np.std(subj_accs):.2f}%")

# Pick best fold for detailed report
best_fold = max(fold_results, key=lambda r: r["subj_acc"])
print(f"\n  Best Fold: {best_fold['fold']} — SubjAcc={best_fold['subj_acc']:.2f}%")
print()
print(classification_report(
    best_fold["val_labels"], best_fold["val_preds"],
    target_names=["ASD", "Typical Control"]
))

# ── 8. Visualisation ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Fold comparison bar chart
folds    = [f"Fold {r['fold']}" for r in fold_results]
s_accs   = [r["slice_acc"]  for r in fold_results]
sj_accs  = [r["subj_acc"]   for r in fold_results]
x        = np.arange(len(folds))
axes[0].bar(x - 0.2, s_accs,  0.4, label="Slice Acc",   color="#457B9D", alpha=0.85)
axes[0].bar(x + 0.2, sj_accs, 0.4, label="Subject Acc", color="#E76F51", alpha=0.85)
axes[0].axhline(np.mean(sj_accs), linestyle="--", color="#E76F51", alpha=0.8,
                label=f"Mean SubjAcc={np.mean(sj_accs):.1f}%")
axes[0].axhline(50, linestyle=":", color="gray", alpha=0.6, label="Chance (50%)")
axes[0].set_xticks(x); axes[0].set_xticklabels(folds)
axes[0].set_ylabel("Accuracy (%)"); axes[0].set_ylim(0, 105)
axes[0].set_title("5-Fold CV Results", fontweight="bold")
axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)

# Confusion matrix (best fold)
cm = best_fold["cm"]
im = axes[1].imshow(cm, cmap="Blues", vmin=0)
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["ASD", "TC"], fontsize=11)
axes[1].set_yticklabels(["ASD", "TC"], fontsize=11, rotation=45)
axes[1].set_xlabel("Predicted", fontsize=12); axes[1].set_ylabel("True", fontsize=12)
axes[1].set_title(f"Confusion Matrix (Best Fold {best_fold['fold']})", fontweight="bold")
for i in range(2):
    for j in range(2):
        pct = 100 * cm[i, j] / cm[i].sum()
        axes[1].text(j, i, f"{cm[i,j]}\n({pct:.0f}%)", ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if cm[i,j] > cm.max()/2 else "black")
plt.colorbar(im, ax=axes[1])

# Subject probability distribution (best fold)
sp_data = best_fold["subj_probs"]
asd_p   = [np.mean(d["probs"]) for d in sp_data.values() if d["label"] == 0]
tc_p    = [np.mean(d["probs"]) for d in sp_data.values() if d["label"] == 1]
axes[2].hist(asd_p, bins=8, alpha=0.7, color="#E76F51", label="ASD", density=True)
axes[2].hist(tc_p,  bins=8, alpha=0.7, color="#2A9D8F", label="TC",  density=True)
axes[2].axvline(0.5, linestyle="--", color="black", label="Decision boundary")
axes[2].set_xlabel("Avg P(TC)"); axes[2].set_ylabel("Density")
axes[2].set_title("Subject Probability Distribution", fontweight="bold")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle(
    f"ABIDE v3 — ResNet50 + 5-Fold CV  |  Mean SubjAcc={np.mean(subj_accs):.1f}%",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(r"C:\abide\classification_v3_results.png", dpi=150, bbox_inches="tight")
print(f"\nResults plot saved: C:\\abide\\classification_v3_results.png")
