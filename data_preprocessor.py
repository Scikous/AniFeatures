import sqlite3
import csv
import os
import shutil
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def db_to_csv(sqlite_location, csv_filename):

    conn = sqlite3.connect(sqlite_location)
    # Define database path and desired columns
    columns_to_extract = ["md5", "file_ext","tag_string"]  # Replace with your desired columns

    # Connect to the database
    cursor = conn.cursor()

    # Build the SQL query to select data
    #query = f"SELECT {', '.join(columns_to_extract)} FROM posts"  # Replace "your_table_name" with the actual table name
    query = f"SELECT md5 || '.' || file_ext AS filename, tag_string FROM posts;"

    # Execute the query and fetch data
    cursor.execute(query)
    data = cursor.fetchall()

    print(data)
    # Open a CSV file for writing
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        #DO NOT MODIFY HEADER, this is required in the image and tag preprocessing in the pre-training phase
        writer.writerow(["filename","tags"])

        # Write each row of data to the CSV file
        for row in data:
            writer.writerow(row)

    # Close the connection
    conn.close()

    print("Data extraction and CSV creation complete!")


#moves images from downloaded subdirectories to one single directory
def images_mover(source_dir, destination_dir):
    # Define destination directory path (create it if it doesn't exist)
    os.makedirs(destination_dir, exist_ok=True)  # Create destination directory if it doesn't exist

    # Iterate through files
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if ".json" not in filename:
                source_path = os.path.join(root, filename)
                destination_path = os.path.join(destination_dir, filename)

                # Move the file
                shutil.move(source_path, destination_path)
                print(f"Moved: {source_path} -> {destination_path}")
    print("File moving complete!")


# Preprocess the image
def preprocess_image(image_path, eval=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if eval==True:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
    return image

def preprocess_data(tags_file, eval=False):
    tags_df = pd.read_csv('tags.csv')
    image_filenames = tags_df['filename'].values
    tags = tags_df['tags'].apply(lambda x: x.split())

    if eval:
        return image_filenames, list(tags)[0]
    # Binarize tags
    mlb = MultiLabelBinarizer()
    binary_tags = mlb.fit_transform(tags)
    print(mlb.classes_[0])
    return image_filenames, binary_tags, mlb.classes_


def main():
    sqlite_loc = 'DanbooruDownloader-master\\DanbooruDownloader\\bin\\Debug\\net6.0-windows\\dataset\\danbooru.sqlite'
    csv_filename = "extracted_data.csv"
    images_source_dir = "images"#"DanbooruDownloader-master\\DanbooruDownloader\\bin\Debug\\net6.0-windows\\dataset\\images"  # Replace with your actual source directory
    images_destination_dir = "images2" 

    db_to_csv(sqlite_loc, csv_filename)
    #images_mover(images_source_dir, images_destination_dir)

if __name__ == "__main__":
    main()