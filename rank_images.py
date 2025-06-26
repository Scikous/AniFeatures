import torch
import timm
from PIL import Image
from train import SiameseNetwork  # Assuming your SiameseNetwork class is in train.py
import torchvision.transforms as transforms
import os
import glob
from tqdm import tqdm # A library for nice progress bars (pip install tqdm)

# --- Configuration ---
MODEL_PATH = 'best_aesthetic_model.pth'
IMAGE_DIR = 'dataset/images_compare/' # The directory containing images to rank
TOP_K = 10 # How many of the top-ranked images to display at the end

# --- Model & Device Setup ---
MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Load Model ---
print("Loading model...")
model = SiameseNetwork(model_name=MODEL_NAME, pretrained=False)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please make sure the path is correct and the model has been trained.")
    exit()

model.to(DEVICE)
model.eval()
print("Model loaded successfully.")

# --- 2. Create Matching Transform ---
data_config = timm.data.resolve_model_data_config(model.backbone)
input_size = data_config['input_size']
transform = transforms.Compose([
    transforms.Resize(size=input_size[1:], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(size=input_size[1:]),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
])

# --- 3. Find and Score All Images in the Directory ---
def score_directory(image_dir):
    # Find all common image types
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_paths:
        print(f"No images found in directory: {image_dir}")
        return []

    print(f"\nFound {len(image_paths)} images. Scoring now...")
    
    results = []
    # Use tqdm for a nice progress bar
    for path in tqdm(image_paths, desc="Ranking Images"):
        try:
            # Prepare image tensor
            image = Image.open(path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # Get score using the model's single-image forward pass
            with torch.no_grad():
                # We use forward_one which is designed to process a single image
                score_tensor = model.forward_one(image_tensor)
            
            # Store the score (as a float) and the path
            results.append((score_tensor.item(), path))

        except Exception as e:
            print(f"\nSkipping file {path} due to an error: {e}")
    
    return results

# --- 4. Rank and Display Results ---
def main():
    ranked_results = score_directory(IMAGE_DIR)

    if not ranked_results:
        return

    # Sort the list of tuples by score in descending order (highest score first)
    ranked_results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n--- Top {TOP_K} Ranked Images ---")
    for i, (score, path) in enumerate(ranked_results[:TOP_K]):
        print(f"Rank {i+1:2d}: Score = {score:.4f} | Path = {path}")

    # The ultimate winner is the first item in the sorted list
    if ranked_results:
        ultimate_winner_path = ranked_results[0][1]
        ultimate_winner_score = ranked_results[0][0]
        print(f"\n🏆 Ultimate Winner: {ultimate_winner_path} (Score: {ultimate_winner_score:.4f})")


if __name__ == '__main__':
    main()