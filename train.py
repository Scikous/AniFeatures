from model import AnimeDataset, AnimeTagger
from utils import tags_to_txt
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms
from torch.utils.data import DataLoader
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


def anifeatures_trainer(dataloader, num_tags, model_save_name="anime_tagger.pth"):
      # Check CUDA availability and set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define model
    model = AnimeTagger(num_tags).to(device)
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):  # Number of epochs
        print("epoch started")
        model.train()
        running_loss = 0.0
        for images, tags in dataloader:
            images = images.to(device)
            tags = tags.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    # Save the model
    torch.save(model.state_dict(), f'models/{model_save_name}')


def main():
    # Load tags
    tags_file = 'metadata.csv'
    image_filenames, binary_tags, num_tags = preprocess_data(tags_file)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = AnimeDataset(image_filenames, binary_tags, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

    anifeatures_trainer(dataloader, num_tags)
    print(num_tags)

if __name__ == "__main__":
    main()

# with torch.no_grad():
#     out_data = model()
#     print(out_data)

# Evaluation code can be added here to check model performance on a validation set.
