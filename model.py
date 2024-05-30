import torch
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import os
import torchvision.models as models


class AnimeDataset(Dataset):
    def __init__(self, image_filenames, binary_tags, transform=None):
        self.image_filenames = image_filenames
        self.binary_tags = binary_tags
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        #change 'images' to other path if need be
        img_name = os.path.join('images', self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        tags = torch.tensor(self.binary_tags[idx], dtype=torch.float32)
        return image, tags


class AnimeTagger(nn.Module):
    def __init__(self, num_tags):
        super(AnimeTagger, self).__init__()
        self.base_model = models.resnet152(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_tags)

    def forward(self, x):
        return torch.sigmoid(self.base_model(x))