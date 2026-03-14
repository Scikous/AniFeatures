#!/usr/bin/env python3
"""
Test script to rank images from a directory using the trained AestheticScorer model.
Images are scored based on aesthetic quality, with automatic domain detection for unmapped images.

Usage:
    python test_rank_images.py [--directory PATH] [--top N]
    
Options:
    --directory PATH  Directory containing images to rank (default: dataset/images_compare)
    --top N           Number of top-ranked images to display (default: 10)
"""

import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import json
import argparse
from tqdm import tqdm

# Try to import open_clip for auto-detection, fall back gracefully if not available
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("Warning: open_clip not installed. Unmapped images will default to Reality domain (0).")

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = 'best_aesthetic_model.pth'
MODEL_NAME = 'vit_large_patch14_dinov2.lvd142m'
DOMAIN_MAP_FILE = "domain_map_auto.json"
DEFAULT_IMAGE_DIR = "dataset/images_compare"
TOP_K = 10

# ImageNet normalization (matching train_dm.py)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 518

# Domain definitions (matching train_dm.py)
NUM_DOMAINS = 4  # 0: Reality, 1: 2D, 2: 3D, 3: Pixel
DOMAIN_DIM = 32
DROPOUT_RATE = 0.3

def try_cuda():
    """Try to use CUDA, fall back to CPU if unavailable or OOM."""
    if not torch.cuda.is_available():
        return 'cpu'
    
    # Check if there's enough free memory for the model (ViT-Large needs ~2GB+)
    free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # Convert to GB
    if free_memory < 2.5:
        print(f"Warning: Only {free_memory:.2f} GB CUDA memory available. Falling back to CPU.")
        return 'cpu'
    
    return 'cuda'


DEVICE = try_cuda()

# ============================================================================
# Model Definition (copied from train_dm.py)
# ============================================================================
class AestheticScorer(torch.nn.Module):
    """Aesthetic scoring model with domain embeddings."""
    
    def __init__(self, model_name, dropout_rate=0.0, num_domains=4, domain_dim=32):
        super().__init__()
        
        # 1. Backbone (Remove classifier head)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        
        # 2. Domain Embedding
        self.domain_emb = torch.nn.Embedding(num_embeddings=num_domains, embedding_dim=domain_dim)
        
        # 3. Scoring Head
        # Input: Image Features + Domain Embedding
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(num_features + domain_dim, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, 1)  # Outputs a raw scalar score
        )

    def forward_one(self, x, domain_idx):
        """Score a single image given its domain."""
        ftrs = self.backbone(x)                     # [Batch, num_features]
        dom = self.domain_emb(domain_idx)           # [Batch, domain_dim]
        combined = torch.cat([ftrs, dom], dim=1)    # [Batch, num_features + domain_dim]
        return self.head(combined)

    def forward(self, img_a, dom_a, img_b, dom_b):
        """Score two images for comparison (used during training)."""
        score_a = self.forward_one(img_a, dom_a)
        score_b = self.forward_one(img_b, dom_b)
        return score_a.squeeze(-1), score_b.squeeze(-1)


# ============================================================================
# Domain Auto-Detection (using OpenCLIP)
# ============================================================================
CLASS_PROMPTS = {
    0: ["a photo", "a photograph", "realistic photo", "cosplay photo", "real life"],
    1: ["anime", "illustration", "drawing", "digital painting", "2d art", "manga", "cartoon"],
    2: ["3d render", "unreal engine", "blender render", "cgi", "3d model", "video game screenshot"],
    3: ["pixel art", "8-bit", "16-bit", "sprite", "dot art"]
}


def create_domain_classifier():
    """Create an OpenCLIP-based domain classifier."""
    if not OPENCLIP_AVAILABLE:
        return None
    
    device = DEVICE
    print("Loading OpenCLIP ViT-H-14 for domain auto-detection...")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', pretrained='laion2b_s32b_b79k'
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')

    # Pre-compute text embeddings for each class (averaging the prompts)
    class_embeddings = []
    for i in range(NUM_DOMAINS):
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
    
    return model, preprocess, text_features


def auto_detect_domain(image_path, classifier):
    """Auto-detect domain for an image using OpenCLIP."""
    if classifier is None:
        return 0  # Default to Reality
    
    model, preprocess, text_features = classifier
    device = DEVICE
    
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity and get prediction
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred_idx = torch.argmax(similarity, dim=1).cpu().numpy()[0]
        
        return int(pred_idx)
    
    except Exception as e:
        print(f"  Warning: Could not auto-detect domain for {image_path}: {e}")
        return 0  # Default to Reality


# ============================================================================
# Domain Map Loading
# ============================================================================
def load_domain_map():
    """Load the domain map from JSON file."""
    if os.path.exists(DOMAIN_MAP_FILE):
        with open(DOMAIN_MAP_FILE, 'r') as f:
            return json.load(f)
    print(f"Warning: {DOMAIN_MAP_FILE} not found. All images will use auto-detection or default to Reality (0).")
    return {}


# ============================================================================
# Image Scoring
# ============================================================================
def get_image_transform():
    """Create the image transform matching training configuration."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_image_paths(image_dir):
    """Get all image paths from a directory."""
    # Support common image formats including AVIF
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.avif', '*.gif')
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    return image_paths


def score_images(model, image_paths, domain_map, classifier):
    """Score all images and return results."""
    transform = get_image_transform()
    
    if not image_paths:
        print(f"No images found in directory.")
        return []
    
    print(f"\nFound {len(image_paths)} images. Scoring now...")
    
    results = []
    
    for path in tqdm(image_paths, desc="Ranking Images"):
        try:
            # Determine domain
            if path in domain_map:
                domain = domain_map[path]
            else:
                # Auto-detect domain for unmapped images
                domain = auto_detect_domain(path, classifier)
            
            domain_tensor = torch.tensor([domain], dtype=torch.long).to(DEVICE)
            
            # Load and transform image
            image = Image.open(path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # Get score using the model's single-image forward pass
            with torch.no_grad():
                score_tensor = model.forward_one(image_tensor, domain_tensor)
            
            # Store the score (as a float) and the path
            results.append({
                'score': score_tensor.item(),
                'path': path,
                'domain': domain
            })
        
        except Exception as e:
            print(f"\n  Skipping file {path} due to error: {e}")
    
    return results


# ============================================================================
# Results Display
# ============================================================================
def display_results(ranked_results, top_k):
    """Display the top-ranked images."""
    if not ranked_results:
        print("No results to display.")
        return
    
    # Sort by score in descending order (highest score first)
    ranked_results.sort(key=lambda x: x['score'], reverse=True)
    
    domain_names = {0: 'Reality', 1: '2D', 2: '3D', 3: 'Pixel'}
    
    print(f"\n{'='*70}")
    print(f"Top {top_k} Ranked Images")
    print(f"{'='*70}")
    
    display_count = min(top_k, len(ranked_results))
    for i in range(display_count):
        result = ranked_results[i]
        rank = i + 1
        score = result['score']
        path = os.path.basename(result['path'])
        domain = domain_names.get(result['domain'], 'Unknown')
        
        # Create a visual bar for the score (normalized roughly to -2 to 2 range)
        normalized_score = max(-2, min(2, score))
        bar_length = int((normalized_score + 2) / 4 * 30)  # Map to 0-30 chars
        bar = '█' * bar_length
        
        print(f"Rank {rank:2d}: Score = {score:+8.4f} | [{domain:6s}] {bar} {path}")
    
    # Show the ultimate winner
    if ranked_results:
        winner = ranked_results[0]
        print(f"\n{'='*70}")
        print(f"🏆 Ultimate Winner:")
        print(f"   Path: {winner['path']}")
        print(f"   Score: {winner['score']:+.4f}")
        print(f"   Domain: {domain_names.get(winner['domain'], 'Unknown')}")
        print(f"{'='*70}")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Rank images from a directory using the trained AestheticScorer model."
    )
    parser.add_argument('--directory', '-d', type=str, default=DEFAULT_IMAGE_DIR,
                        help=f'Directory containing images to rank (default: {DEFAULT_IMAGE_DIR})')
    parser.add_argument('--top', '-n', type=int, default=TOP_K,
                        help=f'Number of top-ranked images to display (default: {TOP_K})')
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return
    
    # Load model
    print("="*60)
    print("Aesthetic Image Ranker")
    print("="*60)
    print(f"\nDevice: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Loading model from '{MODEL_PATH}'...")
    
    try:
        model = AestheticScorer(MODEL_NAME, DROPOUT_RATE, NUM_DOMAINS, DOMAIN_DIM).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please ensure the model has been trained and saved.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(DEVICE)
    model.eval()
    
    # Load domain map
    print(f"\nLoading domain map from '{DOMAIN_MAP_FILE}'...")
    domain_map = load_domain_map()
    print(f"Loaded {len(domain_map)} image-domain mappings.")
    
    # Create domain classifier for auto-detection
    classifier = None
    unmapped_count = 0
    
    # Get image paths
    image_paths = get_image_paths(args.directory)
    
    if not image_paths:
        print(f"No images found in '{args.directory}'.")
        return
    
    # Count unmapped images to decide whether to load OpenCLIP
    for path in image_paths:
        if path not in domain_map:
            unmapped_count += 1
    
    if unmapped_count > 0 and OPENCLIP_AVAILABLE:
        classifier = create_domain_classifier()
        print(f"Auto-detection enabled for {unmapped_count} unmapped images.")
    elif unmapped_count > 0:
        print(f"Warning: {unmapped_count} images not in domain map. Will default to Reality (0).")
    
    # Score all images
    results = score_images(model, image_paths, domain_map, classifier)
    
    if not results:
        print("No valid results obtained.")
        return
    
    # Display top-ranked images
    display_results(results, args.top)


if __name__ == '__main__':
    main()
