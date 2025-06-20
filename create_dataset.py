import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import csv
import shutil

# --- Configuration ---
# TODO: Update these paths before running the script
SOURCE_DIR = "./dataset/images_unlabeled"  # Directory containing the images to be labeled
PROCESSED_DIR = "./dataset/images"  # Directory where labeled images will be moved
CSV_FILE = "labels.csv"      # Name of the CSV file to store the labels

class ImageLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeler")

        # Create processed directory if it doesn't exist
        if not os.path.exists(PROCESSED_DIR):
            os.makedirs(PROCESSED_DIR)

        self.image_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if len(self.image_files) < 2:
            messagebox.showinfo("Info", "Not enough images to form a pair for labeling.")
            self.root.quit()
            return

        self.current_pair_index = 0
        self.create_widgets()
        self.load_next_pair()

        # Bind arrow keys
        self.root.bind("<Left>", lambda event: self.process_label(-1.0))
        self.root.bind("<Right>", lambda event: self.process_label(1.0))
        self.root.bind("<Down>", lambda event: self.process_label(0.0))

    def create_widgets(self):
        # Frames for images
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10)

        self.left_image_label = tk.Label(self.image_frame)
        self.left_image_label.pack(side=tk.LEFT, padx=10)

        self.right_image_label = tk.Label(self.image_frame)
        self.right_image_label.pack(side=tk.RIGHT, padx=10)

        # Frame for buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)

        self.left_button = tk.Button(self.button_frame, text="Left (-1.0)", command=lambda: self.process_label(-1.0))
        self.left_button.pack(side=tk.LEFT, padx=5)

        self.down_button = tk.Button(self.button_frame, text="Down (0.0)", command=lambda: self.process_label(0.0))
        self.down_button.pack(side=tk.LEFT, padx=5)

        self.right_button = tk.Button(self.button_frame, text="Right (1.0)", command=lambda: self.process_label(1.0))
        self.right_button.pack(side=tk.LEFT, padx=5)

    def load_next_pair(self):
        if self.current_pair_index >= len(self.image_files):
            messagebox.showinfo("Info", "All images have been labeled.")
            self.root.quit()
            return

        self.img1_name = self.image_files[self.current_pair_index]
        self.img2_name = self.image_files[self.current_pair_index + 1]

        self.img1_path = os.path.join(SOURCE_DIR, self.img1_name)
        self.img2_path = os.path.join(SOURCE_DIR, self.img2_name)

        # Display images
        img1 = Image.open(self.img1_path)
        img1.thumbnail((800, 800))
        self.photo1 = ImageTk.PhotoImage(img1)
        self.left_image_label.config(image=self.photo1)

        img2 = Image.open(self.img2_path)
        img2.thumbnail((800, 800))
        self.photo2 = ImageTk.PhotoImage(img2)
        self.right_image_label.config(image=self.photo2)

    def process_label(self, label):
        # Write to CSV
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.img1_path, self.img2_path, label])

        # Move files
        shutil.move(self.img1_path, os.path.join(PROCESSED_DIR, self.img1_name))
        shutil.move(self.img2_path, os.path.join(PROCESSED_DIR, self.img2_name))

        # Load next pair
        self.current_pair_index += 2
        
        # Update the list of available images
        self.image_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if self.current_pair_index >= len(self.image_files):
             messagebox.showinfo("Info", "All images have been labeled.")
             self.root.quit()
             return

        self.load_next_pair()


if __name__ == "__main__":
    # --- Setup Directories and CSV ---
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        # You can add some dummy images here for testing
        # For example:
        # Image.new('RGB', (100, 100), color = 'red').save(os.path.join(SOURCE_DIR, 'img1.png'))
        # Image.new('RGB', (100, 100), color = 'blue').save(os.path.join(SOURCE_DIR, 'img2.png'))
        # Image.new('RGB', (100, 100), color = 'green').save(os.path.join(SOURCE_DIR, 'img3.png'))
        # Image.new('RGB', (100, 100), color = 'yellow').save(os.path.join(SOURCE_DIR, 'img4.png'))

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image1_path', 'image2_path', 'label'])


    root = tk.Tk()
    app = ImageLabeler(root)
    root.mainloop()