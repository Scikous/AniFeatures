import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

# --- Configuration ---
CSV_PATH = 'labels.csv'
MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m'
BATCH_SIZE = 4 # Reduced slightly to be safer with validation overhead
LEARNING_RATE = 1e-5
EPOCHS = 50 # The maximum number of epochs to run
MARGIN = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- NEW: Early Stopping & Validation Configuration ---
VALIDATION_SPLIT = 0.2 # Use 20% of the data for validation
PATIENCE = 2 # Stop training if val loss doesn't improve for 5 epochs
BEST_MODEL_PATH = 'best_aesthetic_model.pth'

# --- 1. Custom Dataset Class (No changes needed) ---
class PreferenceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.preferences_df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.preferences_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_a_path = self.preferences_df.iloc[idx]['image1_path']
        img_b_path = self.preferences_df.iloc[idx]['image2_path']
        
        if not os.path.exists(img_a_path):
            raise FileNotFoundError(f"Image not found at {img_a_path}")
        if not os.path.exists(img_b_path):
            raise FileNotFoundError(f"Image not found at {img_b_path}")
            
        image_a = Image.open(img_a_path).convert('RGB')
        image_b = Image.open(img_b_path).convert('RGB')
        
        label = self.preferences_df.iloc[idx]['label']
        
        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
            
        return image_a, image_b, torch.tensor(label, dtype=torch.float32)

# --- 2. Siamese Network Architecture (No changes needed) ---
class SiameseNetwork(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=True):
        super(SiameseNetwork, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        num_ftrs = self.backbone.num_features
        self.backbone.head = nn.Linear(num_ftrs, 1)

    def forward_one(self, x):
        return self.backbone(x)

    def forward(self, image_a, image_b):
        score_a = self.forward_one(image_a)
        score_b = self.forward_one(image_b)
        return score_a.squeeze(), score_b.squeeze()

# --- 3. The Training and Validation Loop (HEAVILY UPDATED) ---
def train_model():
    print(f"Using device: {DEVICE}")

    # --- Data Loading and Splitting ---
    full_df = pd.read_csv(
        CSV_PATH,
        dtype={'image1_path': str, 'image2_path': str, 'label': float}
    )
    # Split the dataframe into training and validation sets
    train_df, val_df = train_test_split(full_df, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"Data split: {len(train_df)} training samples, {len(val_df)} validation samples.")

    # --- Model and Transforms ---
    model = SiameseNetwork().to(DEVICE)
    data_config = timm.data.resolve_model_data_config(model.backbone)
    input_size = data_config['input_size']
    
    transform = transforms.Compose([
        transforms.Resize(size=input_size[1:], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size=input_size[1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])

    # Create separate datasets and dataloaders for training and validation
    train_dataset = PreferenceDataset(dataframe=train_df, transform=transform)
    val_dataset = PreferenceDataset(dataframe=val_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Loss and Optimizer ---
    criterion = nn.MarginRankingLoss(margin=MARGIN)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- Checkpointing & Early Stopping Variables ---
    best_val_loss = np.inf
    epochs_no_improve = 0

    print(f"\nStarting training for a maximum of {EPOCHS} epochs...")
    print(f"Early stopping patience set to {PATIENCE} epochs.")

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for image_a, image_b, labels in train_loader:
            image_a, image_b, labels = image_a.to(DEVICE), image_b.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            score_a, score_b = model(image_a, image_b)
            loss = criterion(score_a, score_b, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * image_a.size(0)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad(): # No need to compute gradients during validation
            for image_a, image_b, labels in val_loader:
                image_a, image_b, labels = image_a.to(DEVICE), image_b.to(DEVICE), labels.to(DEVICE)
                score_a, score_b = model(image_a, image_b)
                loss = criterion(score_a, score_b, labels)
                running_val_loss += loss.item() * image_a.size(0)

        # --- Epoch Summary & Checkpointing ---
        avg_train_loss = running_train_loss / len(train_dataset)
        avg_val_loss = running_val_loss / len(val_dataset)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpoint the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Validation loss improved. Saving best model to {BEST_MODEL_PATH}")
        else:
            epochs_no_improve += 1
            print(f"  -> Validation loss did not improve. Patience: {epochs_no_improve}/{PATIENCE}")

        # Early stopping condition
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    print('\nFinished Training.')
    print(f"The best model (Val Loss: {best_val_loss:.4f}) is saved at {BEST_MODEL_PATH}")

if __name__ == '__main__':
    train_model()