from model import AnimeDataset, AnimeTagger
from utils import tags_to_txt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def preprocess_data(tags_file):
    tags_df = pd.read_csv(tags_file)
    image_filenames = tags_df['filename'].values
    tags = tags_df['tags'].apply(lambda x: x.split())

    # Binarize tags
    mlb = MultiLabelBinarizer()
    binary_tags = mlb.fit_transform(tags)
    print(len(mlb.classes_),mlb.classes_)
    tags_to_txt(mlb.classes_)
    return image_filenames, binary_tags, len(mlb.classes_)


def anifeatures_trainer(train_loader, val_loader, num_tags, model_save_name="anime_tagger.pth"):
      # Check CUDA availability and set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define model
    model = AnimeTagger(num_tags).to(device)
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #######new
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(10):  # Number of epochs
        print(f"Epoch {epoch+1} started")
        model.train()
        running_loss = 0.0
        for images, tags in train_loader:
            images = images.to(device)
            tags = tags.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, tags in val_loader:
                images = images.to(device)
                tags = tags.to(device)
                outputs = model(images)
                loss = criterion(outputs, tags)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'models/{model_save_name}')
            print(f"Model saved with Validation Loss: {best_val_loss}")


def main():
    # Load tags
    tags_file = 'metadata.csv'
    image_filenames, binary_tags, num_tags = preprocess_data(tags_file)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Split data into training and validation sets
    train_indices, val_indices = train_test_split(range(len(image_filenames)), test_size=0.2, random_state=42)
    train_dataset = Subset(AnimeDataset(image_filenames, binary_tags, transform=transform), train_indices)
    val_dataset = Subset(AnimeDataset(image_filenames, binary_tags, transform=transform), val_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=2)

    anifeatures_trainer(train_loader, val_loader, num_tags, model_save_name="anime_tagger3.pth")

    # dataset = AnimeDataset(image_filenames, binary_tags, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

    # anifeatures_trainer(dataloader, num_tags)
    print(num_tags)

if __name__ == "__main__":
    main()

# with torch.no_grad():
#     out_data = model()
#     print(out_data)

# Evaluation code can be added here to check model performance on a validation set.
