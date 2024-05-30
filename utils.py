import os
from PIL import Image
from torchvision import transforms


def tags_to_txt(tags):
    with open('tags.txt', mode='w', encoding='utf-8') as file:
        # Write tags used in training to file
        for tag in tags:
            file.write(tag+'\n')

    print("finished tagging")

def tags_getter(tags_file):
    with open(tags_file, mode='r',encoding='utf-8') as file:
        # Write the data
        tags = file.read()
        tags = tags.splitlines()
        #print(tags)
    return tags

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


def check_file_exists(filepath):
  """Checks if a file exists at the given path."""
  return os.path.exists(filepath)


