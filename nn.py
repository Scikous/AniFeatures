#1
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
#2
import torch.nn as nn
import torchvision.models as models

#3
import torch.optim as optim

# Load tags
tags_df = pd.read_csv('tags2.csv')
image_filenames = tags_df['filename'].values
tags = tags_df['tags'].apply(lambda x: x.split())
# Binarize tags
mlb = MultiLabelBinarizer()
binary_tags = mlb.fit_transform(tags)
#print(binary_tags, tags)

class AnimeDataset(Dataset):
    def __init__(self, image_filenames, binary_tags, transform=None):
        self.image_filenames = image_filenames
        self.binary_tags = binary_tags
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join('images', self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        tags = torch.tensor(self.binary_tags[idx], dtype=torch.float32)
        return image, tags

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = AnimeDataset(image_filenames, binary_tags, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



class AnimeTagger(nn.Module):
    def __init__(self, num_tags):
        super(AnimeTagger, self).__init__()
        self.base_model = models.resnet152(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_tags)
    
    def forward(self, x):
        return torch.sigmoid(self.base_model(x))

num_tags = len(mlb.classes_)
model = AnimeTagger(num_tags)
print(num_tags)


# #####

# # Define loss and optimizer
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# for epoch in range(10):  # Number of epochs
#     model.train()
#     running_loss = 0.0
#     for images, tags in dataloader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, tags)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

# # Save the model
# torch.save(model.state_dict(), 'anime_tagger.pth')


# Load the model for evaluation
model.load_state_dict(torch.load('anime_tagger.pth'))
model.eval()


# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Make predictions
def predict_tags(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = outputs.squeeze().numpy()
        predicted_tags = [mlb.classes_[i] for i, prob in enumerate(probabilities) if prob > threshold]
    return predicted_tags
image_path = "images\\84a168e2e0d0814bcc9665be6dac1cf4.png"

# Preprocess the image
image_tensor = preprocess_image(image_path)

predicted_tags = predict_tags(model, image_tensor)
print("Predicted tags:", predicted_tags)

# with torch.no_grad():
#     out_data = model()
#     print(out_data)

# Evaluation code can be added here to check model performance on a validation set.
