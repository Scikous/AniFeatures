# AniFeatures
This project allows you to train a deep learning model to understand **your** personal aesthetic taste. Instead of relying on generic "good" or "bad" labels, you train the model by making direct A-vs-B comparisons. The final model can then score and rank any collection of images, acting as your personalized art curator.

## How It Works

The core challenge in aesthetic ranking is its subjectivity. A standard classifier struggles because "beauty" isn't an objective label like "cat" or "dog." This project overcomes that challenge using a **Siamese Network** architecture for **preference learning**.

1.  **The Problem with Classification:** Simply classifying images as "good" or "bad" is too coarse. The model can't learn the subtle reasons *why* you might prefer one masterpiece over another.

2.  **The Siamese Solution:** A Siamese Network uses two identical backbones with shared weights to process two images simultaneously. Instead of predicting a class for each, it learns to output a single "aesthetic score".

3.  **Training with Preferences:** We train the network on pairs of images. When you say you prefer Image A over Image B, the model learns to adjust its weights so that `score(A)` is higher than `score(B)`. This is achieved using a **Margin Ranking Loss**, which directly optimizes for this relative difference.

4.  **A Powerful Backbone:** We use a state-of-the-art **Vision Transformer (ViT)** model (`vit_large_patch14_dinov2`) as the backbone. Unlike older CNNs, ViTs can capture global relationships between different parts of an image, making them exceptionally well-suited for understanding abstract concepts like composition and balance.

5.  **The Result:** The final model is a powerful feature extractor fine-tuned to your specific taste. It can assign a meaningful score to any image, allowing you to rank large collections with high accuracy.

## Project Structure

```
.
├── dataset/
│   ├── images_unlabeled/     # -> Place new images for labeling here
│   └── images/               # -> Labeled images are moved here automatically
├── create_dataset.py         # GUI tool for creating preference labels
├── train.py                  # Script to train the Siamese network
├── rank_images.py            # Script to rank a directory of images using the trained model
├── best_aesthetic_model.pth  # -> The final trained model weights (output of train.py)
├── labels.csv                # -> The preference data (output of create_dataset.py)
└── README.md                 # This file
```

## Setup and Installation

### 1. Clone the Repository
```bash
git clone <this-repository-url>
cd <AniFeatures>
```

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
Install all the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage Workflow

The project follows a simple three-step process: **Create Data -> Train Model -> Rank Images**.

### Step 1: Create Your Dataset

This is the most important step. The quality of your model depends entirely on the quality and quantity of your labeled data.

1.  **Populate the Directory:** Place all the images you want to use for training into the `dataset/images_unlabeled/` folder.
2.  **Run the Labeling Tool:** Execute the `create_dataset.py` script.
    ```bash
    python create_dataset.py
    ```
3.  **Start Labeling:** A GUI window will appear showing two images. Use the buttons or arrow keys to record your preference:
    *   **Left Arrow / Left Button:** You prefer the left image (`-1.0`).
    *   **Right Arrow / Right Button:** You prefer the right image (`1.0`).
    *   **Down Arrow / Down Button:** The images are of equal quality (`0.0`). *Note: It's often better to force a choice to give the model a stronger signal.*

As you label, the choices are saved to `labels.csv`, and the processed images are moved from `dataset/images_unlabeled/` to `dataset/images/` to avoid re-labeling. Continue until you have at least a few hundred pairs (more is always better!).

### Step 2: Train the Model

Once you have a `labels.csv` file with sufficient data, you can train your personalized model.

1.  **Verify Configuration (Optional):** Open `train.py`. The script is configured to use `labels.csv` by default. You can adjust hyperparameters like `BATCH_SIZE`, `LEARNING_RATE`, and `EPOCHS` if needed. A powerful GPU (like an RTX 4090) is recommended.
2.  **Run the Training Script:**
    ```bash
    python train.py
    ```
3.  **Wait for Training to Complete:** The script will load the ViT model, fine-tune it on your preference data, and print the loss at each epoch. When finished, it will save the model's weights to a file named `aesthetic_siamese_model.pth`.
    *   **Pro-tip:** Rename this file to something descriptive, like `best_aesthetic_model.pth`.

### Step 3: Rank New Images

With your trained model, you can now find the best images in any collection.

1.  **Configure the Ranking Script:** Open `rank_images.py`.
    *   Set `MODEL_PATH` to the name of your saved model file (e.g., `'best_aesthetic_model.pth'`).
    *   Set `IMAGE_DIR` to the path of the folder containing the images you want to rank.
2.  **Run the Ranking Script:**
    ```bash
    python rank_images.py
    ```
3.  **View the Results:** The script will process every image in the directory, assign it an aesthetic score, and print a sorted list of the top images, with the ultimate winner at the very end.

    **Example Output:**
    ```
    --- Top 10 Ranked Images ---
    Rank  1: Score = 2.7314 | Path = dataset/unlabeled/masterpiece_1.jpg
    Rank  2: Score = 2.5881 | Path = dataset/unlabeled/amazing_art_5.png
    Rank  3: Score = 2.1409 | Path = dataset/unlabeled/great_2.jpeg
    ...

    🏆 Ultimate Winner: dataset/unlabeled/masterpiece_1.jpg (Score: 2.7314)
    ```

You can now use this tool to discover your favorite pieces from massive collections of art, wallpapers, or photos.