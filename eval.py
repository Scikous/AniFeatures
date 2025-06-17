import torch
import timm
from PIL import Image
from train import SiameseNetwork
# Assume the SiameseNetwork class definition is available from the training script

# --- Configuration ---
MODEL_PATH = 'aesthetic_siamese_model.pth'
IMG_PATH_1 = 'path/to/your/art1.jpg'
IMG_PATH_2 = 'path/to/your/art2.jpg'
MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Load Model and Transforms ---
model = SiameseNetwork(model_name=MODEL_NAME, pretrained=False) # Not loading timm pretrained weights
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval() # Set model to evaluation mode

data_config = timm.data.resolve_model_data_config(model.backbone)
transform = timm.data.create_transform(**data_config)

# --- Prepare Images ---
def prepare_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0) # Add batch dimension

image1_tensor = prepare_image(IMG_PATH_1).to(DEVICE)
image2_tensor = prepare_image(IMG_PATH_2).to(DEVICE)

# --- Get Scores ---
with torch.no_grad(): # No need to calculate gradients for inference
    score1, score2 = model(image1_tensor, image2_tensor)

print(f"Image 1 Score: {score1.item():.4f}")
print(f"Image 2 Score: {score2.item():.4f}")

if score1.item() > score2.item():
    print("\nResult: The model predicts you will prefer Image 1.")
else:
    print("\nResult: The model predicts you will prefer Image 2.")