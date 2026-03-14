import torch
import timm
from PIL import Image
import os
import glob
from tqdm import tqdm
import pandas as pd
import json
from train_dm import AestheticScorer as SiameseNetwork
import torchvision.transforms as transforms

# --- Configuration ---
MODEL_PATH = 'best_aesthetic_model.pth'
UNLABELED_DIR = './dataset/all_images'
DOMAIN_MAP_FILE = 'domain_map_auto.json'
OUTPUT_CSV = 'smart_pairs_queue.csv' # We will save the pairs here

# MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m'
MODEL_NAME = 'vit_base_patch16_224.augreg_in21k_ft_in1k'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print("--- 1. Loading Model ---")
    model = SiameseNetwork(model_name=MODEL_NAME)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("Error: Model not found. Train your model first!")
        return
    model.to(DEVICE)
    model.eval()

    # Setup Transforms
    data_config = timm.data.resolve_model_data_config(model.backbone)
    input_size = data_config['input_size']
    transform = transforms.Compose([
        transforms.Resize(size=input_size[1:], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size=input_size[1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])

    # Load Domain Map
    try:
        with open(DOMAIN_MAP_FILE, 'r') as f:
            domain_map = json.load(f)
    except FileNotFoundError:
        domain_map = {}

    print("--- 2. Finding Images ---")
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.avif', '*.gif')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(UNLABELED_DIR, ext)))

    if not image_paths:
        print("No unlabeled images found.")
        return

    print(f"Found {len(image_paths)} images. Scoring...")

    # --- 3. Score All Images ---
    scored_images = []

    with torch.no_grad():
        for path in tqdm(image_paths):
            try:
                # Handle Domain
                domain = domain_map.get(path, 0) # Default to 0 (Reality) if unknown
                domain_tensor = torch.tensor([domain], dtype=torch.long).to(DEVICE)

                # Handle Image
                img = Image.open(path)
                if path.lower().endswith('.gif'):
                    try: img.seek(img.n_frames // 2)
                    except: pass
                img = img.convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)

                # Inference
                score = model.forward_one(img_tensor, domain_tensor).item()
                scored_images.append({'path': path, 'score': score})
            except Exception as e:
                print(f"Error processing {path}: {e}")

    # --- 4. Sort and Pair ---
    print("--- 4. Generating Smart Pairs ---")
    # Sort by score (High to Low)
    scored_images.sort(key=lambda x: x['score'], reverse=True)

    pairs = []
    # Pair neighbor with neighbor: 0&1, 2&3, 4&5...
    for i in range(0, len(scored_images) - 1, 2):
        img_a = scored_images[i]['path']
        img_b = scored_images[i+1]['path']
        # We just save the paths. The labeler will handle the rest.
        pairs.append([os.path.basename(img_a), os.path.basename(img_b)])

    # Save to CSV
    df = pd.DataFrame(pairs, columns=['img1', 'img2'])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(pairs)} smart pairs to {OUTPUT_CSV}")
    print("You can now update create_dataset.py to read from this file.")

if __name__ == '__main__':
    main()
