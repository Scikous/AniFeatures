import csv
import os
from pathlib import Path

def check_dataset_images(csv_filepath):
    # Sets to store absolute paths for accurate 1-to-1 comparison
    images_in_csv = set()
    image_directories = set()

    # Lists to keep track of discrepancies
    missing_from_dir =[]
    missing_from_csv =[]

    # Valid image extensions to prevent reading metadata files (.txt, .json, etc.)
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}

    print("1. Reading CSV and checking if images exist in the directory...")

    # Read the CSV file
    with open(csv_filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        for line_num, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue # Skip empty or malformed rows

            # Extract the first two columns (the image paths)
            img1_path = Path(row[0].strip())
            img2_path = Path(row[1].strip())

            for img_path in (img1_path, img2_path):
                abs_path = img_path.resolve() # Convert to absolute path

                # Record the path and its parent directory
                images_in_csv.add(abs_path)
                image_directories.add(abs_path.parent)

                # Check if the image from the CSV actually exists on your hard drive
                if not abs_path.exists():
                    missing_from_dir.append((line_num, str(img_path)))

    # Display images that are in the CSV but missing from the folder
    if missing_from_dir:
        print("\n[!] Found images in the CSV that DO NOT exist in the directory:")
        for line_num, path in missing_from_dir:
            print(f"   Row {line_num}: {path}")
    else:
        print("\n[+] Success: All images listed in the CSV exist in the directories.")

    print("\n2. Scanning directories for images that are NOT in the CSV...")

    # Loop through the directories we identified from the CSV
    for directory in image_directories:
        if not directory.exists():
            continue

        # Loop through every file in the directory
        for file_path in directory.iterdir():
            # Check if the file is actually an image
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                abs_file_path = file_path.resolve()

                # Check if the file found in the directory is missing from our CSV set
                if abs_file_path not in images_in_csv:
                    # Save the relative path for cleaner console output
                    missing_from_csv.append(file_path)

    # Display images that are in the folder but missing from the CSV
    if missing_from_csv:
        print("\n[!] The following images exist in the directory but DO NOT exist in the CSV:")
        for path in missing_from_csv:
            print(f"   {path}")
    else:
        print("\n[+] Success: No extra images found. Directory and CSV match perfectly.")

if __name__ == "__main__":
    # ---> Change this to the path of your CSV file <---
    CSV_FILE_PATH = 'labels.csv'

    if os.path.exists(CSV_FILE_PATH):
        check_dataset_images(CSV_FILE_PATH)
    else:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
