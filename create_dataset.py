# import tkinter as tk
# from tkinter import messagebox
# from PIL import Image, ImageTk
# import os, sys
# import csv
# import shutil

# # --- Configuration ---
# # TODO: Update these paths before running the script
# SOURCE_DIR = "./dataset/images_unlabeled"  # Directory containing the images to be labeled
# PROCESSED_DIR = "./dataset/images"  # Directory where labeled images will be moved
# CSV_FILE = "labels.csv"      # Name of the CSV file to store the labels



# def validate_and_clean_unlabeled_dir():
#     """
#     Checks if any images in the SOURCE_DIR already exist in the PROCESSED_DIR.
#     If duplicates are found, it prompts the user to automatically move them.
#     """
#     print("--- Running Data Integrity Check ---")
    
#     # Ensure both directories exist to avoid errors
#     if not os.path.exists(SOURCE_DIR) or not os.path.exists(PROCESSED_DIR):
#         print("Source or processed directory does not exist. Skipping check.")
#         return

#     # Get filenames from both directories using sets for efficient comparison
#     unlabeled_files = set(os.listdir(SOURCE_DIR))
#     processed_files = set(os.listdir(PROCESSED_DIR))

#     # Find the intersection (i.e., the duplicate files)
#     duplicates = unlabeled_files.intersection(processed_files)

#     if not duplicates:
#         print("✅ No duplicate images found. Ready for labeling.")
#         return

#     print(f"⚠️ Found {len(duplicates)} image(s) in '{SOURCE_DIR}' that have already been processed:")
#     for f in list(duplicates)[:10]: # Print up to 10 examples
#         print(f" - {f}")
#     if len(duplicates) > 10:
#         print("   ...")

#     # Ask the user for confirmation to clean up the directory
#     response = input(f"\nDo you want to automatically move these {len(duplicates)} files to the '{PROCESSED_DIR}' directory? (y/n): ").lower()

#     if response == 'y':
#         print("Moving duplicate files...")
#         for filename in duplicates:
#             src_path = os.path.join(SOURCE_DIR, filename)
#             dst_path = os.path.join(PROCESSED_DIR, filename)
#             shutil.move(src_path, dst_path)
#             print(f"Moved: {filename}")
#         print("✅ Cleanup complete.")
#     else:
#         print("\nAborting. Please manually clean the 'images_unlabeled' directory before running the labeler again.")
#         sys.exit() # Exit the script to prevent labeling with duplicates



# class ImageLabeler:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Image Labeler")

#         # Create processed directory if it doesn't exist
#         if not os.path.exists(PROCESSED_DIR):
#             os.makedirs(PROCESSED_DIR)

#         self.image_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

#         if len(self.image_files) < 2:
#             messagebox.showinfo("Info", "Not enough images to form a pair for labeling.")
#             self.root.quit()
#             return

#         self.current_pair_index = 0
#         self.create_widgets()
#         self.load_next_pair()

#         # Bind arrow keys
#         self.root.bind("<Left>", lambda event: self.process_label(-1.0))
#         self.root.bind("<Right>", lambda event: self.process_label(1.0))
#         self.root.bind("<Down>", lambda event: self.process_label(0.0))

#     def create_widgets(self):
#         # Frames for images
#         self.image_frame = tk.Frame(self.root)
#         self.image_frame.pack(pady=10)

#         self.left_image_label = tk.Label(self.image_frame)
#         self.left_image_label.pack(side=tk.LEFT, padx=10)

#         self.right_image_label = tk.Label(self.image_frame)
#         self.right_image_label.pack(side=tk.RIGHT, padx=10)

#         # Frame for buttons
#         self.button_frame = tk.Frame(self.root)
#         self.button_frame.pack(pady=10)

#         self.left_button = tk.Button(self.button_frame, text="Left (-1.0)", command=lambda: self.process_label(-1.0))
#         self.left_button.pack(side=tk.LEFT, padx=5)

#         self.down_button = tk.Button(self.button_frame, text="Down (0.0)", command=lambda: self.process_label(0.0))
#         self.down_button.pack(side=tk.LEFT, padx=5)

#         self.right_button = tk.Button(self.button_frame, text="Right (1.0)", command=lambda: self.process_label(1.0))
#         self.right_button.pack(side=tk.LEFT, padx=5)

#     def load_next_pair(self):
#         if self.current_pair_index >= len(self.image_files):
#             messagebox.showinfo("Info", "All images have been labeled.")
#             self.root.quit()
#             return

#         self.img1_name = self.image_files[self.current_pair_index]
#         self.img2_name = self.image_files[self.current_pair_index + 1]

#         self.img1_path = os.path.join(SOURCE_DIR, self.img1_name)
#         self.img2_path = os.path.join(SOURCE_DIR, self.img2_name)

#         # Display images
#         img1 = Image.open(self.img1_path)
#         img1.thumbnail((800, 800))
#         self.photo1 = ImageTk.PhotoImage(img1)
#         self.left_image_label.config(image=self.photo1)

#         img2 = Image.open(self.img2_path)
#         img2.thumbnail((800, 800))
#         self.photo2 = ImageTk.PhotoImage(img2)
#         self.right_image_label.config(image=self.photo2)

#     def process_label(self, label):
#         # Write to CSV
#         with open(CSV_FILE, 'a', newline='') as f:
#             writer = csv.writer(f)
#             # writer.writerow([self.img1_path, self.img2_path, label])
#             writer.writerow([os.path.join(PROCESSED_DIR, self.img1_name), os.path.join(PROCESSED_DIR, self.img2_name), label])

#         # Move files
#         shutil.move(self.img1_path, os.path.join(PROCESSED_DIR, self.img1_name))
#         shutil.move(self.img2_path, os.path.join(PROCESSED_DIR, self.img2_name))

#         # Load next pair
#         self.current_pair_index += 2
        
#         # Update the list of available images
#         self.image_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

#         if self.current_pair_index >= len(self.image_files):
#              messagebox.showinfo("Info", "All images have been labeled.")
#              self.root.quit()
#              return

#         self.load_next_pair()


# if __name__ == "__main__":
#     # --- Setup Directories and CSV ---
#     if not os.path.exists(SOURCE_DIR):
#         os.makedirs(SOURCE_DIR)
#         # You can add some dummy images here for testing
#         # For example:
#         # Image.new('RGB', (100, 100), color = 'red').save(os.path.join(SOURCE_DIR, 'img1.png'))
#         # Image.new('RGB', (100, 100), color = 'blue').save(os.path.join(SOURCE_DIR, 'img2.png'))
#         # Image.new('RGB', (100, 100), color = 'green').save(os.path.join(SOURCE_DIR, 'img3.png'))
#         # Image.new('RGB', (100, 100), color = 'yellow').save(os.path.join(SOURCE_DIR, 'img4.png'))

#     if not os.path.exists(PROCESSED_DIR):
#         os.makedirs(PROCESSED_DIR)

#     if not os.path.isfile(CSV_FILE):
#         with open(CSV_FILE, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['image1_path', 'image2_path', 'label'])

#     # --- Run Validation Before Starting GUI ---
#     validate_and_clean_unlabeled_dir()

#     root = tk.Tk()
#     app = ImageLabeler(root)
#     root.mainloop()



import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import os
import csv
import shutil
import sys

# --- Configuration ---
SOURCE_DIR = "./dataset/images_unlabeled"
PROCESSED_DIR = "./dataset/images"
CSV_FILE = "labels.csv"
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')

# --- New: Clash Resolution GUI ---
class DuplicateResolverApp:
    def __init__(self, root, clash_files):
        self.root = root
        self.root.title("Resolve Name Clashes")
        self.clash_files = clash_files
        self.current_clash_index = 0

        self.create_widgets()
        self.load_next_clash()

    def create_widgets(self):
        # Info Label
        self.info_label = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.info_label.pack(pady=(10, 0))

        # Main frame for images
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10, padx=10)

        # Left (New) Image
        self.left_frame = tk.Frame(self.image_frame)
        self.left_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(self.left_frame, text="New Image (in 'images_unlabeled')").pack()
        self.left_image_label = tk.Label(self.left_frame)
        self.left_image_label.pack()

        # Right (Existing) Image
        self.right_frame = tk.Frame(self.image_frame)
        self.right_frame.pack(side=tk.RIGHT, padx=10)
        tk.Label(self.right_frame, text="Existing Image (in 'images')").pack()
        self.right_image_label = tk.Label(self.right_frame)
        self.right_image_label.pack()

        # Button Frame
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)

        self.delete_button = tk.Button(self.button_frame, text="Delete New Image", bg="salmon", command=self.delete_new)
        self.delete_button.pack(side=tk.LEFT, padx=20)

        self.rename_button = tk.Button(self.button_frame, text="Keep & Rename New Image", bg="lightgreen", command=self.rename_new)
        self.rename_button.pack(side=tk.RIGHT, padx=20)

    def load_next_clash(self):
        if self.current_clash_index >= len(self.clash_files):
            messagebox.showinfo("Complete", "All name clashes have been resolved.")
            self.root.destroy() # Close this window to proceed
            return

        filename = self.clash_files[self.current_clash_index]
        self.info_label.config(text=f"Resolving clash {self.current_clash_index + 1}/{len(self.clash_files)} for: {filename}")
        
        new_path = os.path.join(SOURCE_DIR, filename)
        existing_path = os.path.join(PROCESSED_DIR, filename)

        # Display new image
        img1 = Image.open(new_path)
        img1.thumbnail((600, 600))
        self.photo1 = ImageTk.PhotoImage(img1)
        self.left_image_label.config(image=self.photo1)

        # Display existing image
        img2 = Image.open(existing_path)
        img2.thumbnail((600, 600))
        self.photo2 = ImageTk.PhotoImage(img2)
        self.right_image_label.config(image=self.photo2)

    def delete_new(self):
        filename = self.clash_files[self.current_clash_index]
        path_to_delete = os.path.join(SOURCE_DIR, filename)
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete\n{filename}\nfrom the 'images_unlabeled' directory?"):
            try:
                os.remove(path_to_delete)
                print(f"Deleted: {path_to_delete}")
                self.current_clash_index += 1
                self.load_next_clash()
            except Exception as e:
                messagebox.showerror("Error", f"Could not delete file: {e}")

    def rename_new(self):
        original_filename = self.clash_files[self.current_clash_index]
        original_path = os.path.join(SOURCE_DIR, original_filename)
        
        new_filename = simpledialog.askstring("Rename File", "Enter the new filename:", initialvalue=original_filename)

        if not new_filename:
            return # User cancelled

        new_path = os.path.join(SOURCE_DIR, new_filename)

        if os.path.exists(new_path) or os.path.exists(os.path.join(PROCESSED_DIR, new_filename)):
            messagebox.showerror("Error", "A file with this new name already exists in either the source or processed directory. Please choose another name.")
            return

        try:
            os.rename(original_path, new_path)
            print(f"Renamed: '{original_filename}' -> '{new_filename}'")
            self.current_clash_index += 1
            self.load_next_clash()
        except Exception as e:
            messagebox.showerror("Error", f"Could not rename file: {e}")

# --- Main Labeling GUI (No changes needed inside this class) ---
class ImageLabeler:
    # ... (This class remains exactly as it was) ...
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeler")
        self.image_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(self.image_files) < 2:
            messagebox.showinfo("Info", "Not enough images to form a pair for labeling.")
            self.root.quit()
            return
        self.current_pair_index = 0
        self.create_widgets()
        self.load_next_pair()
        self.root.bind("<Left>", lambda event: self.process_label(-1.0))
        self.root.bind("<Right>", lambda event: self.process_label(1.0))
        self.root.bind("<Down>", lambda event: self.process_label(0.0))
    def create_widgets(self):
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10)
        self.left_image_label = tk.Label(self.image_frame)
        self.left_image_label.pack(side=tk.LEFT, padx=10)
        self.right_image_label = tk.Label(self.image_frame)
        self.right_image_label.pack(side=tk.RIGHT, padx=10)
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)
        self.left_button = tk.Button(self.button_frame, text="Left is Better", command=lambda: self.process_label(-1.0))
        self.left_button.pack(side=tk.LEFT, padx=5)
        self.equal_button = tk.Button(self.button_frame, text="Equal", command=lambda: self.process_label(0.0))
        self.equal_button.pack(side=tk.LEFT, padx=5)
        self.right_button = tk.Button(self.button_frame, text="Right is Better", command=lambda: self.process_label(1.0))
        self.right_button.pack(side=tk.LEFT, padx=5)
    def load_next_pair(self):
        if self.current_pair_index + 1 >= len(self.image_files):
            messagebox.showinfo("Info", "All available images have been processed.")
            self.root.quit()
            return
        self.img1_name, self.img2_name = self.image_files[self.current_pair_index], self.image_files[self.current_pair_index + 1]
        self.img1_path, self.img2_path = os.path.join(SOURCE_DIR, self.img1_name), os.path.join(SOURCE_DIR, self.img2_name)
        img1 = Image.open(self.img1_path); img1.thumbnail((800, 800)); self.photo1 = ImageTk.PhotoImage(img1)
        self.left_image_label.config(image=self.photo1)
        img2 = Image.open(self.img2_path); img2.thumbnail((800, 800)); self.photo2 = ImageTk.PhotoImage(img2)
        self.right_image_label.config(image=self.photo2)
    def process_label(self, label):
        path1 = os.path.join(PROCESSED_DIR, self.img1_name); path2 = os.path.join(PROCESSED_DIR, self.img2_name)
        with open(CSV_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([path1, path2, label])
        shutil.move(self.img1_path, path1); shutil.move(self.img2_path, path2)
        self.current_pair_index += 2
        self.image_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.load_next_pair()


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 1. Setup Directories and CSV ---
    for path in [SOURCE_DIR, PROCESSED_DIR]:
        if not os.path.exists(path):
            os.makedirs(path)
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            f.write('image1_path,image2_path,label\n')
    
    # --- 2. Find and Resolve Name Clashes ---
    print("--- Checking for potential name clashes ---")
    unlabeled_files = set(f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(VALID_EXTENSIONS))
    processed_files = set(f for f in os.listdir(PROCESSED_DIR) if f.lower().endswith(VALID_EXTENSIONS))
    
    clashes = list(unlabeled_files.intersection(processed_files))

    if clashes:
        print(f"⚠️ Found {len(clashes)} potential name clash(es). Starting resolver GUI...")
        resolver_root = tk.Tk()
        DuplicateResolverApp(resolver_root, clashes)
        resolver_root.mainloop() # This blocks execution until the resolver window is closed
        print("✅ Clash resolution complete.")
    else:
        print("✅ No name clashes found.")

    # --- 3. Start the Main Labeling Application ---
    print("--- Starting Image Labeler ---")
    main_root = tk.Tk()
    app = ImageLabeler(main_root)
    main_root.mainloop()