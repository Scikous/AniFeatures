## image labeler automatic
import os
import json
import torch
from PIL import Image
import open_clip
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = "./dataset/all_images"
DOMAIN_MAP_FILE = "domain_map_auto.json"
BATCH_SIZE = 32

# 0: Reality, 1: 2D Illust, 2: 3D Render, 3: Pixel Art
# We use multiple prompts per category to help the model understand nuances
CLASS_PROMPTS = {
    0: ["a photo", "a photograph", "realistic photo", "cosplay photo", "real life"],
    1: ["anime", "illustration", "drawing", "digital painting", "2d art", "manga", "cartoon"],
    2: ["3d render", "unreal engine", "blender render", "cgi", "3d model", "video game screenshot"],
    3: ["pixel art", "8-bit", "16-bit", "sprite", "dot art"]
}

def load_domain_map():
    if os.path.exists(DOMAIN_MAP_FILE):
        with open(DOMAIN_MAP_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_domain_map(data):
    with open(DOMAIN_MAP_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ViT-H-14 for Tagging on {device}...")
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')

    # Pre-compute text embeddings for each class (averaging the prompts)
    class_embeddings = []
    for i in range(4):
        prompts = CLASS_PROMPTS[i]
        text = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # Average the features for this class
            mean_feature = text_features.mean(dim=0)
            mean_feature /= mean_feature.norm()
            class_embeddings.append(mean_feature)
    
    # Stack into a single tensor [4, 1024]
    text_features = torch.stack(class_embeddings)

    # Load Map
    domain_map = load_domain_map()
    all_files = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    unlabeled_files = [f for f in all_files if f not in domain_map]
    
    print(f"Tagging {len(unlabeled_files)} images...")

    for i in tqdm(range(0, len(unlabeled_files), BATCH_SIZE)):
        batch_paths = unlabeled_files[i : i + BATCH_SIZE]
        images = []
        valid_paths = []

        for path in batch_paths:
            try:
                image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
                images.append(image)
                valid_paths.append(path)
            except:
                continue

        if not images: continue

        image_input = torch.cat(images).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            preds = torch.argmax(similarity, dim=1).cpu().numpy()

        for path, pred_idx in zip(valid_paths, preds):
            domain_map[path] = int(pred_idx)

        if i % (BATCH_SIZE * 10) == 0:
            save_domain_map(domain_map)

    save_domain_map(domain_map)
    print("Tagging complete.")

if __name__ == "__main__":
    main()