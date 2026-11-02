import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from tqdm.auto import tqdm

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== ABIDE Brain MRI Classification Pipeline ===")
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ─── 1. Load Phenotypic Labels ───
df = pd.read_csv(r"C:\abide\Phenotypic_V1_0b.csv")
dx_map = {}
for _, row in df.iterrows():
    s = str(row['SUB_ID'])
    dx = int(row['DX_GROUP']) - 1
    dx_map[s] = dx
    dx_map[s.zfill(7)] = dx
print(f"\nLoaded {len(df)} subject labels from CSV")

# ─── 2. Collect Slices with Labels ───
anat_dir = r"C:\abide\OUTPUT\anat"
raw_records = []
for subj in os.listdir(anat_dir):
    if subj == "UCLA_1": continue
    sp = os.path.join(anat_dir, subj)
    if not os.path.isdir(sp): continue
    label = dx_map.get(subj, dx_map.get(subj.lstrip('0'), None))
    if label is None: continue
    for s in glob.glob(os.path.join(sp, "*.png")):
        raw_records.append({"path": s, "subject_id": subj, "label": label})

print(f"Total raw slices: {len(raw_records)}")

# ─── 3. Brain Slice Filtering ───
print("\nFiltering empty/non-brain slices...")
filtered_records = []
for record in tqdm(raw_records, desc="Filtering"):
    try:
        arr = np.array(Image.open(record["path"]).convert("L"))
        if arr.std() >= 5.0 and (arr > 15).mean() >= 0.025:
            filtered_records.append(record)
    except:
        pass

asd_c = sum(1 for r in filtered_records if r["label"] == 0)
tc_c  = sum(1 for r in filtered_records if r["label"] == 1)
print(f"Retained {len(filtered_records)} brain slices | ASD={asd_c}  TC={tc_c}  Ratio={asd_c/tc_c:.2f}")

# ─── 4. Subject-Level Stratified Split ───
subjects = [r["subject_id"] for r in filtered_records]
labels   = [r["label"]      for r in filtered_records]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_idx = next(sgkf.split(np.zeros(len(filtered_records)), labels, subjects))

train_records = [filtered_records[i] for i in train_idx]
val_records   = [filtered_records[i] for i in val_idx]

train_subjs = set(r["subject_id"] for r in train_records)
val_subjs   = set(r["subject_id"] for r in val_records)
print(f"\nTrain: {len(train_records)} slices from {len(train_subjs)} subjects")
print(f"Val  : {len(val_records)} slices from {len(val_subjs)} subjects")
print(f"Subject overlap (should be 0): {len(train_subjs & val_subjs)}")

# ─── 5. Compute Class Weights ───
n_total = len(train_records)
n_asd   = sum(1 for r in train_records if r["label"] == 0)
n_tc    = sum(1 for r in train_records if r["label"] == 1)
w_asd   = n_total / (2 * n_asd)
w_tc    = n_total / (2 * n_tc)
class_weights = torch.tensor([w_asd, w_tc], dtype=torch.float32, device=device)
print(f"\nClass Weights -> ASD: {w_asd:.4f}  TC: {w_tc:.4f}")

# ─── 6. Dataset with Per-Scan Z-Score Normalization ───
class ABIDESliceDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        # Load grayscale and apply per-scan z-score normalization
        arr = np.array(Image.open(item["path"]).convert("L"), dtype=np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)   # z-score per scan
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0).astype(np.uint8)
        image = Image.fromarray(arr).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, item["label"], item["subject_id"]

# MRI-appropriate augmentations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),          # Brain is roughly symmetric L/R
    transforms.RandomRotation(10),              # Simulate slight head tilt
    transforms.RandomAffine(                    # Simulate different head positions
        degrees=5,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Batch size = 32
# Reasoning: 10,231 train slices / 32 = ~320 steps/epoch
# More gradient updates per epoch vs batch=64
# Better generalization on small 73-subject dataset
# Large batches average out gradients → miss fine-grained patterns
BATCH_SIZE = 32

train_dataset = ABIDESliceDataset(train_records, transform=train_transform)
val_dataset   = ABIDESliceDataset(val_records,   transform=val_transform)

# NO WeightedRandomSampler — risk of overfitting on 32 TC subjects
# (showing same 32 TC brains 1.14x more → model memorizes those exact brains)
# Class imbalance is handled entirely by class_weights in the loss function
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0, pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available()
)
print(f"Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")

# ─── 7. Build DenseNet121 Model ───
weights = models.DenseNet121_Weights.IMAGENET1K_V1
model = models.densenet121(weights=weights)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Unfreeze DenseBlock4 + BatchNorm5 for domain adaptation
for param in model.features.denseblock4.parameters():
    param.requires_grad = True
for param in model.features.norm5.parameters():
    param.requires_grad = True

num_features = model.classifier.in_features

# Custom classification head with two Dropout layers
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)

model = model.to(device)
print("\nModel loaded on:", device)

# ─── 8. Loss, Optimizer, Scheduler ───
# ── Epoch Planning ──────────────────────────────────────────────
# Dataset : 10,231 train slices | 2,558 val slices
# Batch   : 32 → ~320 steps/epoch
# Strategy: 3-epoch warmup → cosine decay → early stopping patience=7
# Max epochs: 30 (early stopping will likely trigger at ~15-20)
EPOCHS     = 30
WARMUP     = 3   # LR linearly rises 1e-6 → 2e-4 over 3 epochs
LR         = 2e-4
PATIENCE   = 7   # Stop if val subject-acc doesn't improve for 7 epochs

criterion  = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer  = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)

# LR Warmup: prevents destroying pretrained DenseBlock4 weights with large early updates
def lr_lambda(current_epoch):
    if current_epoch < WARMUP:
        return float(current_epoch + 1) / float(WARMUP)  # linear warmup
    # Cosine decay after warmup
    progress = (current_epoch - WARMUP) / float(max(1, EPOCHS - WARMUP))
    return max(1e-6 / LR, 0.5 * (1.0 + np.cos(np.pi * progress)))

scheduler  = LambdaLR(optimizer, lr_lambda=lr_lambda)
scaler     = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
print(f"Epochs: {EPOCHS} (max) | Warmup: {WARMUP} epochs | Early stopping patience: {PATIENCE}")

# ─── 9. Training & Validation Functions ───
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            # Gradient clipping: prevents explosive updates that cause underfitting recovery oscillation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total   += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    subject_probs = {}
    with torch.no_grad():
        for images, labels, subjs in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for i, subj in enumerate(subjs):
                if subj not in subject_probs:
                    subject_probs[subj] = {"probs": [], "label": labels[i].item()}
                subject_probs[subj]["probs"].append(probs[i, 1].item())

    slice_acc = accuracy_score(all_labels, all_preds) * 100.0
    subj_correct = sum(
        1 for d in subject_probs.values()
        if (1 if np.mean(d["probs"]) >= 0.5 else 0) == d["label"]
    )
    subj_acc = 100.0 * subj_correct / len(subject_probs)
    return running_loss / len(loader.dataset), slice_acc, subj_acc, all_preds, all_labels

# ─── 10. Training Loop ───
best_subj_acc = 0.0
best_epoch    = 0
no_improve    = 0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "slice_acc": [], "subj_acc": [], "lr": []}

print("\n" + "="*60)
print("Starting Training...")
print(f"  Batch size: {BATCH_SIZE} | Steps/epoch: {len(train_loader)}")
print(f"  Max epochs: {EPOCHS} | LR warmup: {WARMUP} epochs | Early stop patience: {PATIENCE}")
print("="*60)

for epoch in range(EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
    val_loss, slice_acc, subj_acc, _, _ = validate(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["slice_acc"].append(slice_acc)
    history["subj_acc"].append(subj_acc)
    history["lr"].append(current_lr)

    overfit_gap = train_acc - slice_acc
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] LR={current_lr:.2e} | Train: {train_loss:.4f}/{train_acc:.1f}% | Val: {val_loss:.4f}/{slice_acc:.1f}% | SubjAcc: {subj_acc:.1f}% | Gap: {overfit_gap:.1f}%")

    if subj_acc > best_subj_acc:
        best_subj_acc = subj_acc
        best_epoch    = epoch + 1
        no_improve    = 0
        torch.save(model.state_dict(), r"C:\abide\best_abide_densenet121.pth")
        print(f"  >>> Best model saved! Subject Acc: {best_subj_acc:.2f}% at epoch {best_epoch}")
    else:
        no_improve += 1
        print(f"  No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch+1}!")
            break

    # Overfitting warning
    if overfit_gap > 15:
        print(f"  [WARNING] Overfit warning: train-val gap = {overfit_gap:.1f}%")

print(f"\nTraining Complete! Best Subject-Level Accuracy: {best_subj_acc:.2f}% at epoch {best_epoch}")

# ─── 11. Final Evaluation ───
if os.path.exists(r"C:\abide\best_abide_densenet121.pth"):
    model.load_state_dict(torch.load(r"C:\abide\best_abide_densenet121.pth"))
_, slice_acc, subj_acc, val_preds, val_labels = validate(model, val_loader, criterion)
class_names = ["Autistic (ASD)", "Typical Control (TC)"]
print("\n=== FINAL CLASSIFICATION REPORT ===")
print(classification_report(val_labels, val_preds, target_names=class_names))
cm = confusion_matrix(val_labels, val_preds)

# Save evaluation plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history["train_acc"], label="Train Acc")
axes[0].plot(history["slice_acc"], label="Val Slice Acc")
axes[0].plot(history["subj_acc"],  label="Val Subject Acc", linestyle="--")
axes[0].set_title("Accuracy History")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].imshow(cm, cmap="Blues")
axes[1].set_title("Confusion Matrix")
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(class_names); axes[1].set_yticklabels(class_names)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", color="red", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(r"C:\abide\classification_evaluation.png", dpi=150)
print(f"\nEvaluation plot saved to C:\\abide\\classification_evaluation.png")
print(f"\nFinal Subject Accuracy: {subj_acc:.2f}%")
