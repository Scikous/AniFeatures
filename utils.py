import csv
import os
import shutil
import sqlite3
from PIL import Image
from torchvision import transforms

# Write tags used in training to file
def tags_to_txt(tags):
    with open('tags.txt', mode='w', encoding='utf-8') as file:
        for tag in tags:
            file.write(tag+'\n')

    print("finished tagging")

#read tags from .txt file and return as a list
def tags_getter(tags_file):
    with open(tags_file, mode='r',encoding='utf-8') as file:
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

#used to make sure a file can be deleted
def check_file_exists(filepath):
  """Checks if a file exists at the given path."""
  return os.path.exists(filepath)

def process_images_in_directory(source_dir, image_processor):
    """
    Helper function to iterate through images in a directory and apply a processing function.

    Parameters:
    source_dir (str): Source directory containing the images.
    image_processor (function): Function to process each image.
    """
    for root, _, images in os.walk(source_dir):
        for img in images:
            if img.endswith('.png') or img.endswith('.jpg'):
                img_path = os.path.join(root, img)
                image_processor(img_path)

def image_loader(source_dir):
    imgs = []

    def add_to_list(img_path):
        imgs.append(img_path)

    process_images_in_directory(source_dir, add_to_list)
    print("Images loading complete!")
    return imgs

def image_mover(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    def move_image(img_path):
        shutil.move(img_path, destination_dir)
        print(f"Moved: {img_path} -> {destination_dir}")

    process_images_in_directory(source_dir, move_image)
    print("File moving complete!")

def image_validator(source_dir):
    def validate_image(img_path):
        try:
            with Image.open(img_path) as image:
                image = image.convert('RGB')
        except OSError as e:
            if "truncated" in str(e):
                print(f"Error: Image at {img_path} is corrupted. Deleting...")
                os.remove(img_path)

    process_images_in_directory(source_dir, validate_image)
    print("Finished deleting all broken images")


def db_to_csv(sqlite_location, csv_filename):

    conn = sqlite3.connect(sqlite_location)
    # Connect to the database
    cursor = conn.cursor()

    # Build the SQL query to select data
    #query = f"SELECT {', '.join(columns_to_extract)} FROM posts"  # Replace "your_table_name" with the actual table name
    query = f"SELECT md5 || '.' || file_ext AS filename, tag_string FROM posts;"

    # Execute the query and fetch data
    cursor.execute(query)
    data = cursor.fetchall()

    # Open a CSV file for writing
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        #DO NOT MODIFY HEADER, this is required in the image and tag preprocessing in the pre-training phase
        writer.writerow(["filename","tags"])

        # Write each row of data to the CSV file
        for row in data:
            #can't make use of .swf and prob other formats. file formats
            if check_file_exists('images/'+row[0]) and (row[0].endswith('.png') or row[0].endswith('.jpg') or row[0].endswith('.gif')):
                writer.writerow(row)

    # Close the connection
    conn.close()

    print("Data extraction and CSV creation complete!")


