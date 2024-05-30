import sqlite3
import csv
import os
import shutil
from PIL import Image

from utils import check_file_exists

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

    # Open a CSV file for writing
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        #DO NOT MODIFY HEADER, this is required in the image and tag preprocessing in the pre-training phase
        writer.writerow(["filename","tags"])

        # Write each row of data to the CSV file
        for row in data:
            
            #can't make use of .gif or .swf, etc. file formats
            if check_file_exists('images/'+row[0]) and (row[0].endswith('.png') or row[0].endswith('.jpg')):
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
            if filename.endswith('.png') or filename.endswith('.jpg'):
                source_path = os.path.join(root, filename)
                destination_path = os.path.join(destination_dir, filename)

                # Move the file
                shutil.move(source_path, destination_path)
                print(f"Moved: {source_path} -> {destination_path}")
    print("File moving complete!")

def image_validator(source_dir):
    for root,_,images in os.walk(source_dir):
        print(len(images))
        for img in images:
            if img.endswith('.jpg') or img.endswith('.png'):
                img_path = os.path.join(root, img)
                try:
                    with Image.open(img_path) as image:
                        image = image.convert('RGB')
                except OSError as e:
                    if "truncated" in str(e):
                        print(f"Error: Image {img} is corrupted. Deleting...")
                        os.remove(img_path)

    print("Finished deleting all broken images")

def main():
    sqlite_loc = 'DanbooruDownloader-master\\DanbooruDownloader\\bin\\Debug\\net6.0-windows\\dataset\\danbooru.sqlite'
    csv_filename = "metadata.csv"
    images_source_dir = "images"#"DanbooruDownloader-master\\DanbooruDownloader\\bin\Debug\\net6.0-windows\\dataset\\images"  # Replace with your actual source directory
    images_destination_dir = "images2" 

    image_validator('images')
    db_to_csv(sqlite_loc, csv_filename)
    images_mover(images_source_dir, images_destination_dir)

if __name__ == "__main__":
    main()