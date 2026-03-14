
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.transforms as transforms
import timm
import pandas as pd
from PIL import Image
import pillow_avif  # Ensures AVIF support works
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# --- Configuration ---
CSV_PATH = 'labels.csv'
# Using a slightly more robust model for aesthetic judgement
# MODEL_NAME = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m'
# --- Hyperparameters ---
BATCH_SIZE = 4         # Increased slightly; 2 is very noisy for batch norm/optimization
LEARNING_RATE = 2e-5    # Low LR for fine-tuning
EPOCHS = 50
WEIGHT_DECAY = 1e-2     # Standard ViT weight decay
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 7
NUM_DOMAINS = 4         # 0: Reality, 1: 2D, 2: 3D, 3: Pixel

# Standard ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Transforms ---
def get_transforms(input_size):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        # Aesthetic judgement relies on color/sharpness, so be careful with Jitter
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transform, val_transform

# --- 2. Dataset ---
class PreferenceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Images
        try:
            img_a = Image.open(row['image1_path']).convert('RGB')
            img_b = Image.open(row['image2_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return dummy tensors to prevent crash, or handle gracefully
            return torch.zeros(3, 518, 518), torch.zeros(3, 518, 518), 0, 0, 0.0

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        # Parse Domains (Ensure they are Ints within 0-3 range)
        d1 = int(row['domain1'])
        d2 = int(row['domain2'])

        # Safety clamp to prevent CUDA assertions
        d1 = max(0, min(d1, NUM_DOMAINS - 1))
        d2 = max(0, min(d2, NUM_DOMAINS - 1))

        domain_a = torch.tensor(d1, dtype=torch.long)
        domain_b = torch.tensor(d2, dtype=torch.long)

        # Label: 1.0 (Right/B Better), -1.0 (Left/A Better)
        label = torch.tensor(float(row['label']), dtype=torch.float32)

        return img_a, img_b, domain_a, domain_b, label

# --- 3. The Model ---
class AestheticScorer(nn.Module):
    def __init__(self, model_name, dropout_rate=0.0, num_domains=4, domain_dim=32):
        super().__init__()

        # 1. Backbone (Remove classifier head)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features

        # 2. Domain Embedding
        self.domain_emb = nn.Embedding(num_embeddings=num_domains, embedding_dim=domain_dim)

        # 3. Scoring Head
        # Input: Image Features + Domain Embedding
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features + domain_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1) # Outputs a raw scalar score
        )

    def forward_one(self, x, domain_idx):
        ftrs = self.backbone(x)                     # [Batch, num_features]
        dom = self.domain_emb(domain_idx)           # [Batch, domain_dim]
        combined = torch.cat([ftrs, dom], dim=1)    # [Batch, num_features + domain_dim]
        return self.head(combined)

    def forward(self, img_a, dom_a, img_b, dom_b):
        score_a = self.forward_one(img_a, dom_a)
        score_b = self.forward_one(img_b, dom_b)
        return score_a.squeeze(-1), score_b.squeeze(-1)

# --- 4. Training Engine ---
def train_model():
    print(f"--- Initialization on {DEVICE} ---")

    # Data Setup
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("CSV not found.")
        return

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Transforms (224 is standard for ViT)
    train_tf, val_tf = get_transforms(518)

    train_ds = PreferenceDataset(train_df, train_tf)
    val_ds = PreferenceDataset(val_df, val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model Setup
    model = AestheticScorer(MODEL_NAME, DROPOUT_RATE, NUM_DOMAINS).to(DEVICE)

    # --- CRITICAL: Optimizer Parameter Grouping ---
    # We want the Backbone to learn slowly (fine-tune) and the Head/Embeds to learn fast.
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters()) + list(model.domain_emb.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1}, # 10x smaller LR for backbone
        {'params': head_params, 'lr': LEARNING_RATE}            # Normal LR for new layers
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Loss Function
    # MarginRankingLoss: loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
    # If y=1, x1 should be > x2.
    criterion = nn.MarginRankingLoss(margin=0.2)

    # Training Loop
    best_val_loss = float('inf')
    early_stop_counter = 0

    scaler = torch.amp.GradScaler('cuda') # Mixed Precision for VRAM efficiency

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for img_a, img_b, dom_a, dom_b, label in pbar:
            img_a, img_b = img_a.to(DEVICE), img_b.to(DEVICE)
            dom_a, dom_b = dom_a.to(DEVICE), dom_b.to(DEVICE)
            label = label.to(DEVICE) # Label is 1.0 (Right Better) or -1.0 (Left Better)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                score_a, score_b = model(img_a, dom_a, img_b, dom_b)

                # LOGIC CHECK:
                # If Label = 1.0, user implies Right (B) is better.
                # MarginRankingLoss(x1, x2, target=1) enforces x1 > x2.
                # Therefore, we pass (score_b, score_a, label).
                # If label=1 -> score_b > score_a. Correct.
                # If label=-1 -> score_a > score_b. Correct.
                loss = criterion(score_b, score_a, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for img_a, img_b, dom_a, dom_b, label in val_loader:
                img_a, img_b = img_a.to(DEVICE), img_b.to(DEVICE)
                dom_a, dom_b = dom_a.to(DEVICE), dom_b.to(DEVICE)
                label = label.to(DEVICE)

                score_a, score_b = model(img_a, dom_a, img_b, dom_b)
                loss = criterion(score_b, score_a, label)
                val_loss += loss.item()

                # Accuracy Calculation
                # If score_b > score_a, model thinks Right is better.
                # If label > 0, truth is Right is better.
                pred_right_better = (score_b > score_a).float()
                true_right_better = (label > 0).float()
                correct_preds += (pred_right_better == true_right_better).sum().item()
                total_preds += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_preds / total_preds

        print(f"Stats: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {accuracy:.2%}")

        scheduler.step()

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_aesthetic_model.pth')
            print(">>> Saved Best Model")
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    # Ensure CUDA checks are blocking to catch embedding errors immediately
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    train_model()
