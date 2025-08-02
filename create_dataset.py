import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import os
import csv
import shutil
import sys
import cairosvg
import pillow_avif 
# --- Configuration ---
SOURCE_DIR = "./dataset/images_unlabeled"
PROCESSED_DIR = "./dataset/images"
CSV_FILE = "labels.csv"
# SVG is handled by the pre-processor, so we only need raster formats here.
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.avif')

# --- New: Clash Resolution GUI (No changes needed) ---
class DuplicateResolverApp:
    # ... This class remains exactly as you provided ...
    def __init__(self, root, clash_files):
        self.root = root; self.root.title("Resolve Name Clashes"); self.clash_files = clash_files; self.current_clash_index = 0
        self.create_widgets(); self.load_next_clash()
        self.root.bind("<Left>", lambda event: self.delete_new()); self.root.bind("<Right>", lambda event: self.rename_new())
    def create_widgets(self):
        self.info_label = tk.Label(self.root, text="", font=("Helvetica", 12)); self.info_label.pack(pady=(10, 0))
        self.image_frame = tk.Frame(self.root); self.image_frame.pack(pady=10, padx=10)
        self.left_frame = tk.Frame(self.image_frame); self.left_frame.pack(side=tk.LEFT, padx=10); tk.Label(self.left_frame, text="New Image (in 'images_unlabeled')").pack(); self.left_image_label = tk.Label(self.left_frame); self.left_image_label.pack()
        self.right_frame = tk.Frame(self.image_frame); self.right_frame.pack(side=tk.RIGHT, padx=10); tk.Label(self.right_frame, text="Existing Image (in 'images')").pack(); self.right_image_label = tk.Label(self.right_frame); self.right_image_label.pack()
        self.button_frame = tk.Frame(self.root); self.button_frame.pack(pady=10)
        self.delete_button = tk.Button(self.button_frame, text="Delete New Image", bg="salmon", command=self.delete_new); self.delete_button.pack(side=tk.LEFT, padx=20)
        self.rename_button = tk.Button(self.button_frame, text="Keep & Rename New Image", bg="lightgreen", command=self.rename_new); self.rename_button.pack(side=tk.RIGHT, padx=20)
    def load_next_clash(self):
        if self.current_clash_index >= len(self.clash_files):
            messagebox.showinfo("Complete", "All name clashes have been resolved."); self.root.destroy(); return
        filename = self.clash_files[self.current_clash_index]; self.info_label.config(text=f"Resolving clash {self.current_clash_index + 1}/{len(self.clash_files)} for: {filename}")
        new_path = os.path.join(SOURCE_DIR, filename); existing_path = os.path.join(PROCESSED_DIR, filename)
        img1 = Image.open(new_path); img1.thumbnail((600, 600)); self.photo1 = ImageTk.PhotoImage(img1); self.left_image_label.config(image=self.photo1)
        img2 = Image.open(existing_path); img2.thumbnail((600, 600)); self.photo2 = ImageTk.PhotoImage(img2); self.right_image_label.config(image=self.photo2)
    def delete_new(self):
        filename = self.clash_files[self.current_clash_index]; path_to_delete = os.path.join(SOURCE_DIR, filename)
        try:
            os.remove(path_to_delete); print(f"Deleted: {path_to_delete}"); self.current_clash_index += 1; self.load_next_clash()
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete file: {e}")
    def rename_new(self):
        original_filename = self.clash_files[self.current_clash_index]; original_path = os.path.join(SOURCE_DIR, original_filename)
        new_filename = simpledialog.askstring("Rename File", "Enter the new filename:", initialvalue=original_filename)
        if not new_filename: return
        new_path = os.path.join(SOURCE_DIR, new_filename)
        if os.path.exists(new_path) or os.path.exists(os.path.join(PROCESSED_DIR, new_filename)):
            messagebox.showerror("Error", "A file with this new name already exists. Please choose another name."); return
        try:
            os.rename(original_path, new_path); print(f"Renamed: '{original_filename}' -> '{new_filename}'"); self.current_clash_index += 1; self.load_next_clash()
        except Exception as e:
            messagebox.showerror("Error", f"Could not rename file: {e}")

# --- Main Labeling GUI (No changes needed) ---
class ImageLabeler:
    # ... This class remains exactly as you provided ...
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeler")
        # Now this line will correctly find the converted SVGs (as PNGs) and WEBP files
        self.image_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(VALID_EXTENSIONS)])
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
        self.left_frame = tk.Frame(self.image_frame); self.left_frame.pack(side=tk.LEFT, padx=10, fill="x", expand=True)
        self.right_frame = tk.Frame(self.image_frame); self.right_frame.pack(side=tk.RIGHT, padx=10, fill="x", expand=True)
        self.left_image_label = tk.Label(self.left_frame); self.left_image_label.pack()
        self.left_name_entry = tk.Entry(self.left_frame, state='readonly', relief='flat', readonlybackground='white', fg='black', justify='center'); self.left_name_entry.pack(pady=5, fill='x')
        self.right_image_label = tk.Label(self.right_frame); self.right_image_label.pack()
        self.right_name_entry = tk.Entry(self.right_frame, state='readonly', relief='flat', readonlybackground='white', fg='black', justify='center'); self.right_name_entry.pack(pady=5, fill='x')
        self.button_frame = tk.Frame(self.root); self.button_frame.pack(pady=10)
        self.left_button = tk.Button(self.button_frame, text="Left is Better", command=lambda: self.process_label(-1.0)); self.left_button.pack(side=tk.LEFT, padx=5)
        self.equal_button = tk.Button(self.button_frame, text="Equal", command=lambda: self.process_label(0.0)); self.equal_button.pack(side=tk.LEFT, padx=5)
        self.right_button = tk.Button(self.button_frame, text="Right is Better", command=lambda: self.process_label(1.0)); self.right_button.pack(side=tk.LEFT, padx=5)
    def load_next_pair(self):
        if self.current_pair_index + 1 >= len(self.image_files):
            messagebox.showinfo("Info", "All available images have been processed."); self.root.quit(); return
        self.img1_name, self.img2_name = self.image_files[self.current_pair_index], self.image_files[self.current_pair_index + 1]
        self.img1_path, self.img2_path = os.path.join(SOURCE_DIR, self.img1_name), os.path.join(SOURCE_DIR, self.img2_name)
        img1 = Image.open(self.img1_path); img1.thumbnail((1200, 1200)); self.photo1 = ImageTk.PhotoImage(img1); self.left_image_label.config(image=self.photo1)
        self.left_name_entry.config(state='normal'); self.left_name_entry.delete(0, tk.END); self.left_name_entry.insert(0, self.img1_name); self.left_name_entry.config(state='readonly')
        img2 = Image.open(self.img2_path); img2.thumbnail((1200, 1200)); self.photo2 = ImageTk.PhotoImage(img2); self.right_image_label.config(image=self.photo2)
        self.right_name_entry.config(state='normal'); self.right_name_entry.delete(0, tk.END); self.right_name_entry.insert(0, self.img2_name); self.right_name_entry.config(state='readonly')
    def process_label(self, label):
        path1 = os.path.join(PROCESSED_DIR, self.img1_name); path2 = os.path.join(PROCESSED_DIR, self.img2_name)
        with open(CSV_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([path1, path2, label])
        shutil.move(self.img1_path, path1); shutil.move(self.img2_path, path2)
        self.image_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(VALID_EXTENSIONS)])
        self.current_pair_index = 0 # Start from the beginning of the new list
        self.load_next_pair()


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 1. Setup Directories and CSV ---
    for path in [SOURCE_DIR, PROCESSED_DIR]:
        if not os.path.exists(path): os.makedirs(path)
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f: f.write('image1_path,image2_path,label\n')
    
    # --- NEW: Pre-process SVG files ---
    print("--- Checking for SVG files to convert ---")
    svg_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.svg')]
    if svg_files:
        print(f"Found {len(svg_files)} SVG file(s). Converting to PNG...")
        for svg_filename in svg_files:
            base_name = os.path.splitext(svg_filename)[0]
            svg_path = os.path.join(SOURCE_DIR, svg_filename)
            png_path = os.path.join(SOURCE_DIR, base_name + '.png')
            
            if os.path.exists(png_path):
                print(f"Skipping '{svg_filename}', a PNG with the same name already exists.")
                continue
                
            try:
                cairosvg.svg2png(url=svg_path, write_to=png_path)
                print(f"Converted '{svg_filename}' -> '{base_name}.png'")
                os.remove(svg_path) # Optionally delete the original SVG
            except Exception as e:
                print(f"Error converting '{svg_filename}': {e}")
        print("✅ SVG conversion complete.")
    else:
        print("✅ No SVG files found.")


    # --- 2. Find and Resolve Name Clashes ---
    print("\n--- Checking for potential name clashes ---")
    unlabeled_files = set(f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(VALID_EXTENSIONS))
    processed_files = set(f for f in os.listdir(PROCESSED_DIR) if f.lower().endswith(VALID_EXTENSIONS))
    clashes = list(unlabeled_files.intersection(processed_files))
    if clashes:
        print(f"⚠️ Found {len(clashes)} potential name clash(es). Starting resolver GUI...")
        resolver_root = tk.Tk()
        DuplicateResolverApp(resolver_root, clashes)
        resolver_root.mainloop()
        print("✅ Clash resolution complete.")
    else:
        print("✅ No name clashes found.")

    # --- 3. Start the Main Labeling Application ---
    print("\n--- Starting Image Labeler ---")
    main_root = tk.Tk()
    app = ImageLabeler(main_root)
    main_root.mainloop()
