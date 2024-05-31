import torch
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import os
import torchvision.models as models

#prepares and loads images for training
class AnimeDataset(Dataset):
    def __init__(self, image_filenames, binary_tags, image_src_dir='dataset/images', transform=None):
        self.image_filenames = image_filenames
        self.image_src_dir = image_src_dir
        self.binary_tags = binary_tags
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    #apply transform to resize, normalize and tensorify images and binarized tags 
    def __getitem__(self, idx):
        #change 'images' to other path if need be
        img_name = os.path.join(self.image_src_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        tags = torch.tensor(self.binary_tags[idx], dtype=torch.float32)
        return image, tags

#defines the deep neural network architecture
class AniFeatures(nn.Module):
    def __init__(self, num_tags):
        super(AniFeatures, self).__init__()
        self.base_model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_tags)

    def forward(self, x):
        return torch.sigmoid(self.base_model(x))