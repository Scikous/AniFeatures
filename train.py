import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import pandas as pd
from PIL import Image
import os

# --- Configuration ---
# Path to your CSV file with (image_A, image_B, label)
CSV_PATH = 'preferences.csv'
# Model choice from timm library. DINOv2 pre-trained ViT is excellent.
MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m' # A powerful Vision Transformer
# Training parameters
BATCH_SIZE = 16  # Adjust based on your VRAM. 16 should be fine on a 4090.
LEARNING_RATE = 1e-5 # Use a small learning rate for fine-tuning
EPOCHS = 50
MARGIN = 0.5 # The margin for the loss function
IMG_SIZE = 224 # ViTs are often trained on 224x224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Custom Dataset Class ---
class PreferenceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.preferences_df = pd.read_csv(csv_file, header=None, names=['img_a', 'img_b', 'label'])
        self.transform = transform

    def __len__(self):
        return len(self.preferences_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_a_path = self.preferences_df.iloc[idx, 0]
        img_b_path = self.preferences_df.iloc[idx, 1]
        
        # Check if paths are valid
        if not os.path.exists(img_a_path):
            raise FileNotFoundError(f"Image not found at {img_a_path}")
        if not os.path.exists(img_b_path):
            raise FileNotFoundError(f"Image not found at {img_b_path}")
            
        image_a = Image.open(img_a_path).convert('RGB')
        image_b = Image.open(img_b_path).convert('RGB')
        
        label = self.preferences_df.iloc[idx, 2]
        
        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
            
        return image_a, image_b, torch.tensor(label, dtype=torch.float32)

# --- 2. The Siamese Network Architecture ---
class SiameseNetwork(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=True):
        super(SiameseNetwork, self).__init__()
        # Load the pre-trained model as the backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the number of input features for the model's classifier
        # For ViT, it's model.head.in_features
        num_ftrs = self.backbone.head.in_features
        
        # Replace the classifier head with a single output neuron
        # This neuron will output the "aesthetic score"
        self.backbone.head = nn.Linear(num_ftrs, 1)

    def forward_one(self, x):
        """Passes one image through the backbone."""
        return self.backbone(x)

    def forward(self, image_a, image_b):
        """Passes both images through the shared-weight backbone."""
        score_a = self.forward_one(image_a)
        score_b = self.forward_one(image_b)
        # Squeeze the output to remove the extra dimension [BATCH_SIZE, 1] -> [BATCH_SIZE]
        return score_a.squeeze(), score_b.squeeze()

# --- 3. The Training Loop ---
def train_model():
    print(f"Using device: {DEVICE}")

    # Define image transformations
    # Use the normalization stats recommended for the pre-trained model
    data_config = timm.data.resolve_model_data_config(MODEL_NAME)
    transform = timm.data.create_transform(**data_config)
    
    print("Data Transforms:")
    print(transform)

    # Create dataset and dataloader
    dataset = PreferenceDataset(csv_file=CSV_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize model, loss, and optimizer
    model = SiameseNetwork().to(DEVICE)
    criterion = nn.MarginRankingLoss(margin=MARGIN)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train() # Set model to training mode
        
        for i, (image_a, image_b, labels) in enumerate(dataloader):
            image_a, image_b, labels = image_a.to(DEVICE), image_b.to(DEVICE), labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            score_a, score_b = model(image_a, image_b)
            
            # Calculate loss
            loss = criterion(score_a, score_b, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 50 == 0: # Print every 50 batches
                print(f'Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / len(dataloader)
        print(f'--- End of Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f} ---')

    print('Finished Training')
    
    # Save the trained model weights
    torch.save(model.state_dict(), 'aesthetic_siamese_model.pth')
    print('Model saved to aesthetic_siamese_model.pth')

if __name__ == '__main__':
    train_model()