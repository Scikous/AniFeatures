import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk, ImageOps
import os
import csv
import shutil
import json
import cairosvg
import pillow_avif


# --- Configuration ---
INPUT_PAIRS_FILE = "smart_pairs_queue.csv"         # <--- NEW: Your input CSV with img1,img2
SOURCE_DIR = "./dataset/all_images"
PROCESSED_DIR = "./dataset/images"
CSV_FILE = "labels.csv"
DOMAIN_MAP_FILE = "domain_map_auto.json"
WHITELIST_FILE = "clash_whitelist.json" # <--- NEW CONFIG
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.avif', '.gif')

# --- DISPLAY CONFIGURATION ---
SINGLE_VIEW_SIZE = (1000, 1000)
PAIR_VIEW_SIZE = (900, 900)

# --- Domain Definitions ---
DOMAIN_LABELS = {
    0: "Reality",
    1: "2D Illust",
    2: "3D Render",
    3: "Pixel Art"
}
# Reverse mapping for the dropdown logic
DOMAIN_NAMES_TO_IDS = {v: k for k, v in DOMAIN_LABELS.items()}
DOMAIN_OPTIONS = list(DOMAIN_LABELS.values())

# Color coding for domain labels
DOMAIN_COLORS = {
    "Reality": "blue",
    "2D Illust": "red",
    "3D Render": "pink",
    "Pixel Art": "green"
}

# --- Helper Functions ---
def load_domain_map():
    if os.path.exists(DOMAIN_MAP_FILE):
        with open(DOMAIN_MAP_FILE, 'r') as f: return json.load(f)
    return {}

def save_domain_map(data):
    with open(DOMAIN_MAP_FILE, 'w') as f: json.dump(data, f, indent=4)

# --- NEW WHITELIST HELPER FUNCTIONS ---
def load_whitelist():
    if os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, 'r') as f: return json.load(f)
    return []

def save_whitelist(data):
    with open(WHITELIST_FILE, 'w') as f: json.dump(data, f, indent=4)
# --------------------------------------

def open_image_robust(path):
    try:
        img = Image.open(path)
        if path.lower().endswith('.gif'):
            try:
                total_frames = getattr(img, 'n_frames', 1)
                img.seek(total_frames // 2)
            except Exception: pass
        return img.convert('RGB')
    except Exception as e:
        print(f"Error opening {path}: {e}")
        return Image.new('RGB', (100, 100), color='red')

def create_fixed_photo(img, size):
    img_resized = ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img_resized)

# --- Clash Resolution GUI (UPDATED) ---
class DuplicateResolverApp:
    def __init__(self, root, clash_files):
        self.root = root
        self.root.title("Resolve Name Clashes")
        self.clash_files = clash_files
        self.current_clash_index = 0
        self.whitelist_data = load_whitelist() # Load existing whitelist
        self.create_widgets()
        self.load_next_clash()

        # Key bindings
        self.root.bind("<Left>", lambda event: self.delete_new())
        self.root.bind("<Right>", lambda event: self.rename_new())
        self.root.bind("<Down>", lambda event: self.whitelist_new()) # New binding

    def create_widgets(self):
        self.info_label = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.info_label.pack(pady=(10, 0))

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10, padx=10)

        # Left (New/Source)
        self.left_frame = tk.Frame(self.image_frame)
        self.left_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(self.left_frame, text="New Image (Source)").pack()
        self.left_image_label = tk.Label(self.left_frame)
        self.left_image_label.pack()

        # Right (Existing/Processed)
        self.right_frame = tk.Frame(self.image_frame)
        self.right_frame.pack(side=tk.RIGHT, padx=10)
        tk.Label(self.right_frame, text="Existing Image (Processed)").pack()
        self.right_image_label = tk.Label(self.right_frame)
        self.right_image_label.pack()

        # Buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)

        tk.Button(self.button_frame, text="Delete New (Left)", bg="salmon", command=self.delete_new).pack(side=tk.LEFT, padx=10)
        # New Whitelist Button
        tk.Button(self.button_frame, text="Whitelist / Keep Both (Down)", bg="lightblue", command=self.whitelist_new).pack(side=tk.LEFT, padx=10)
        tk.Button(self.button_frame, text="Rename New (Right)", bg="lightgreen", command=self.rename_new).pack(side=tk.LEFT, padx=10)

    def load_next_clash(self):
        if self.current_clash_index >= len(self.clash_files):
            messagebox.showinfo("Complete", "All name clashes resolved.")
            self.root.destroy()
            return

        filename = self.clash_files[self.current_clash_index]
        self.info_label.config(text=f"Clash: {filename}\n({self.current_clash_index + 1}/{len(self.clash_files)})")

        new_path = os.path.join(SOURCE_DIR, filename)
        existing_path = os.path.join(PROCESSED_DIR, filename)

        # Display images
        img1 = open_image_robust(new_path)
        self.photo1 = create_fixed_photo(img1, PAIR_VIEW_SIZE)
        self.left_image_label.config(image=self.photo1)

        img2 = open_image_robust(existing_path)
        self.photo2 = create_fixed_photo(img2, PAIR_VIEW_SIZE)
        self.right_image_label.config(image=self.photo2)

    def delete_new(self):
        filename = self.clash_files[self.current_clash_index]
        path = os.path.join(SOURCE_DIR, filename)
        try:
            os.remove(path)
            self.current_clash_index += 1
            self.load_next_clash()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def rename_new(self):
        orig_name = self.clash_files[self.current_clash_index]
        orig_path = os.path.join(SOURCE_DIR, orig_name)

        new_name = simpledialog.askstring("Rename", "New filename:", initialvalue=orig_name)
        if not new_name: return

        new_path = os.path.join(SOURCE_DIR, new_name)

        # Check collisions in both Source and Processed
        if os.path.exists(new_path) or os.path.exists(os.path.join(PROCESSED_DIR, new_name)):
            messagebox.showerror("Error", "Name exists in Source or Processed folder.")
            return

        try:
            os.rename(orig_path, new_path)
            self.current_clash_index += 1
            self.load_next_clash()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def whitelist_new(self):
        """
        Adds the original name to whitelist.json,
        then automatically renames the source file with a '_wl' suffix
        so it can exist alongside the processed one.
        """
        filename = self.clash_files[self.current_clash_index]

        # 1. Add to Whitelist JSON (Record keeping)
        if filename not in self.whitelist_data:
            self.whitelist_data.append(filename)
            save_whitelist(self.whitelist_data)

        # 2. Auto-Rename the Source file to resolve the file system conflict
        # Pattern: filename.ext -> filename_wl.ext
        orig_path = os.path.join(SOURCE_DIR, filename)
        name, ext = os.path.splitext(filename)

        new_name = f"{name}_wl{ext}"
        counter = 1
        # Ensure unique name if _wl already exists
        while os.path.exists(os.path.join(SOURCE_DIR, new_name)) or os.path.exists(os.path.join(PROCESSED_DIR, new_name)):
            new_name = f"{name}_wl_{counter}{ext}"
            counter += 1

        try:
            os.rename(orig_path, os.path.join(SOURCE_DIR, new_name))
            print(f"Whitelisted: Renamed '{filename}' to '{new_name}'")
            self.current_clash_index += 1
            self.load_next_clash()
        except Exception as e:
            messagebox.showerror("Error", f"Could not auto-rename for whitelist: {e}")

# --- PHASE 1: Domain Classification App ---
class DomainClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phase 1: Domain Classification (4-Domain System)")
        self.domain_map = load_domain_map()
        all_source_images = {os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.lower().endswith(VALID_EXTENSIONS)}
        self.images_to_classify = sorted(list(all_source_images - set(self.domain_map.keys())))
        self.current_image_index = 0
        if not self.images_to_classify:
            messagebox.showinfo("Info", "No new images to classify.")
            self.root.after(100, self.root.destroy); return
        self.create_widgets()
        self.load_next_image()

    def create_widgets(self):
        self.info_label = tk.Label(self.root, text="", font=("Helvetica", 12)); self.info_label.pack(pady=10)
        self.image_label = tk.Label(self.root); self.image_label.pack(pady=10, padx=10)
        tk.Label(self.root, text="L: Reality | R: 2D | U: 3D | D: Pixel | DEL: Delete", fg="blue").pack(pady=5)
        self.button_frame = tk.Frame(self.root); self.button_frame.pack(pady=10)
        tk.Button(self.button_frame, text="Reality (0) [Left]", width=18, bg="blue", fg="white", command=lambda: self.classify_and_next(0)).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(self.button_frame, text="2D Illust (1) [Right]", width=18, bg="red", fg="white", command=lambda: self.classify_and_next(1)).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.button_frame, text="3D/Render (2) [Up]", width=18, bg="pink", command=lambda: self.classify_and_next(2)).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(self.button_frame, text="Pixel Art (3) [Down]", width=18, bg="green", fg="white", command=lambda: self.classify_and_next(3)).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Delete Image (Del)", bg="#ffcccb", command=self.delete_current_image).pack(pady=10)
        self.root.bind("<Left>", lambda e: self.classify_and_next(0))
        self.root.bind("<Right>", lambda e: self.classify_and_next(1))
        self.root.bind("<Up>", lambda e: self.classify_and_next(2))
        self.root.bind("<Down>", lambda e: self.classify_and_next(3))
        self.root.bind("<Delete>", lambda e: self.delete_current_image())

    def load_next_image(self):
        if self.current_image_index >= len(self.images_to_classify):
            messagebox.showinfo("Complete", "Classification complete."); self.root.destroy(); return
        self.current_path = self.images_to_classify[self.current_image_index]
        self.info_label.config(text=f"Classifying {self.current_image_index + 1}/{len(self.images_to_classify)}\n{os.path.basename(self.current_path)}")
        try:
            img = open_image_robust(self.current_path)
            self.photo = create_fixed_photo(img, SINGLE_VIEW_SIZE)
            self.image_label.config(image=self.photo)
        except Exception as e: self.delete_current_image()

    def classify_and_next(self, domain):
        self.domain_map[self.current_path] = domain
        save_domain_map(self.domain_map)
        self.current_image_index += 1
        self.load_next_image()

    def delete_current_image(self):
        try:
            if os.path.exists(self.current_path): os.remove(self.current_path)
            del self.images_to_classify[self.current_image_index]
            self.load_next_image()
        except Exception as e: messagebox.showerror("Error", str(e))

# --- PHASE 2: Pairwise Labeling App (CSV STRICT SOURCE) ---
class ImageLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Phase 2: Pairwise Labeling (Strict Source)")
        self.domain_map = load_domain_map()

        # Load pairs from the Input CSV
        self.pairs = []
        if os.path.exists(INPUT_PAIRS_FILE):
            try:
                with open(INPUT_PAIRS_FILE, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Detect if header is missing or different, but assuming standard img1,img2 based on prompt
                    for row in reader:
                        # Normalize keys to handle potential whitespace in CSV headers
                        clean_row = {k.strip(): v for k, v in row.items()}
                        i1 = clean_row.get('img1', '').strip()
                        i2 = clean_row.get('img2', '').strip()
                        if i1 and i2:
                            self.pairs.append((i1, i2))
            except Exception as e:
                messagebox.showerror("CSV Error", f"Could not read {INPUT_PAIRS_FILE}:\n{e}")
                self.root.destroy()
                return
        else:
            messagebox.showerror("Missing File", f"Could not find {INPUT_PAIRS_FILE}")
            self.root.destroy()
            return

        if not self.pairs:
            messagebox.showinfo("Info", "No pairs found in CSV."); self.root.quit(); return

        self.current_pair_index = 0
        self.handled_count = 0
        self.create_widgets()
        self.load_next_pair()

        # Bindings
        self.root.bind("<Left>", lambda e: self.process_label(-1.0))
        self.root.bind("<Right>", lambda e: self.process_label(1.0))
        self.root.bind("<Down>", lambda e: self.skip_pair())

    def create_widgets(self):
        self.info_label = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.info_label.pack(pady=10)

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10)

        # --- LEFT SIDE ---
        self.left_frame = tk.Frame(self.image_frame); self.left_frame.pack(side=tk.LEFT, padx=10)
        self.left_image_label = tk.Label(self.left_frame); self.left_image_label.pack()
        self.left_name_entry = tk.Entry(self.left_frame, state='readonly', justify='center'); self.left_name_entry.pack(pady=5, fill='x')

        self.left_color_indicator = tk.Label(self.left_frame, width=3, height=1, bg="gray")
        self.left_color_indicator.pack(side=tk.LEFT, padx=2)
        self.left_domain_var = tk.StringVar(self.root)
        self.left_domain_menu = tk.OptionMenu(self.left_frame, self.left_domain_var, *DOMAIN_OPTIONS, command=lambda val: self.update_domain('left', val))
        self.left_domain_menu.config(width=15); self.left_domain_menu.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Button(self.left_frame, text="Delete Left", bg="#ffcccb", command=lambda: self.delete_image('left')).pack(pady=5)

        # --- RIGHT SIDE ---
        self.right_frame = tk.Frame(self.image_frame); self.right_frame.pack(side=tk.RIGHT, padx=10)
        self.right_image_label = tk.Label(self.right_frame); self.right_image_label.pack()
        self.right_name_entry = tk.Entry(self.right_frame, state='readonly', justify='center'); self.right_name_entry.pack(pady=5, fill='x')

        self.right_color_indicator = tk.Label(self.right_frame, width=3, height=1, bg="gray")
        self.right_color_indicator.pack(side=tk.LEFT, padx=2)
        self.right_domain_var = tk.StringVar(self.root)
        self.right_domain_menu = tk.OptionMenu(self.right_frame, self.right_domain_var, *DOMAIN_OPTIONS, command=lambda val: self.update_domain('right', val))
        self.right_domain_menu.config(width=15); self.right_domain_menu.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Button(self.right_frame, text="Delete Right", bg="#ffcccb", command=lambda: self.delete_image('right')).pack(pady=5)

        # --- BUTTONS ---
        self.button_frame = tk.Frame(self.root); self.button_frame.pack(pady=10)
        tk.Button(self.button_frame, text="Left Better", command=lambda: self.process_label(-1.0)).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Skip (Down)", command=self.skip_pair).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Right Better", command=lambda: self.process_label(1.0)).pack(side=tk.LEFT, padx=5)

    def load_next_pair(self):
        # Recursively find the next valid pair where BOTH images exist in SOURCE_DIR
        while self.current_pair_index < len(self.pairs):
            self.img1_name, self.img2_name = self.pairs[self.current_pair_index]

            # STRICT CHECK: Must be in SOURCE_DIR
            self.img1_path = os.path.join(SOURCE_DIR, self.img1_name)
            self.img2_path = os.path.join(SOURCE_DIR, self.img2_name)

            if os.path.exists(self.img1_path) and os.path.exists(self.img2_path):
                # Both exist, break loop and display
                break
            else:
                # If either is missing (likely moved in previous step), skip silently (or print)
                # print(f"Skipping {self.img1_name} vs {self.img2_name} - One or both missing from Source.")
                self.current_pair_index += 1

        # Check if we ran out of pairs
        if self.current_pair_index >= len(self.pairs):
            messagebox.showinfo("Complete", "All valid pairs in CSV handled.")
            self.root.destroy()
            return

        self.info_label.config(text=f"Pair {self.current_pair_index + 1} / {len(self.pairs)} | Handled: {self.handled_count}")

        # Get current domains
        self.domain1 = self.domain_map.get(self.img1_path, -1)
        self.domain2 = self.domain_map.get(self.img2_path, -1)

        # Update GUI
        domain1_name = DOMAIN_LABELS.get(self.domain1, "Unknown")
        domain2_name = DOMAIN_LABELS.get(self.domain2, "Unknown")
        self.left_domain_var.set(domain1_name)
        self.right_domain_var.set(domain2_name)
        
        # Update color indicators
        self.left_color_indicator.config(bg=DOMAIN_COLORS.get(domain1_name, "gray"))
        self.right_color_indicator.config(bg=DOMAIN_COLORS.get(domain2_name, "gray"))

        try:
            i1 = open_image_robust(self.img1_path)
            self.p1 = create_fixed_photo(i1, PAIR_VIEW_SIZE)
            self.left_image_label.config(image=self.p1)
            self.left_name_entry.config(state='normal'); self.left_name_entry.delete(0, tk.END); self.left_name_entry.insert(0, self.img1_name); self.left_name_entry.config(state='readonly')

            i2 = open_image_robust(self.img2_path)
            self.p2 = create_fixed_photo(i2, PAIR_VIEW_SIZE)
            self.right_image_label.config(image=self.p2)
            self.right_name_entry.config(state='normal'); self.right_name_entry.delete(0, tk.END); self.right_name_entry.insert(0, self.img2_name); self.right_name_entry.config(state='readonly')
        except Exception as e:
            print(f"Error loading images: {e}")
            self.skip_pair()

    def update_domain(self, side, new_value_str):
        new_id = DOMAIN_NAMES_TO_IDS[new_value_str]
        path = self.img1_path if side == 'left' else self.img2_path

        # Update Map
        self.domain_map[path] = new_id
        save_domain_map(self.domain_map)

        if side == 'left':
            self.domain1 = new_id
            self.left_color_indicator.config(bg=DOMAIN_COLORS.get(new_value_str, "gray"))
        else:
            self.domain2 = new_id
            self.right_color_indicator.config(bg=DOMAIN_COLORS.get(new_value_str, "gray"))
        print(f"Updated {side} image domain to: {new_value_str}")

    def process_label(self, label):
        final_p1 = os.path.join(PROCESSED_DIR, self.img1_name)
        final_p2 = os.path.join(PROCESSED_DIR, self.img2_name)

        # 1. Write to CSV
        with open(CSV_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([final_p1, final_p2, self.domain1, self.domain2, label])

        # 2. Update Domain Map (point keys to new location)
        if self.img1_path in self.domain_map:
            self.domain_map[final_p1] = self.domain_map.pop(self.img1_path)
        if self.img2_path in self.domain_map:
            self.domain_map[final_p2] = self.domain_map.pop(self.img2_path)
        save_domain_map(self.domain_map)

        # 3. Move Files (Strict Source -> Processed)
        try:
            shutil.move(self.img1_path, final_p1)
            shutil.move(self.img2_path, final_p2)
        except Exception as e:
            messagebox.showerror("Move Error", f"Error moving files:\n{e}")
            return

        self.handled_count += 1
        self.current_pair_index += 1
        self.load_next_pair()

    def skip_pair(self):
        self.current_pair_index += 1
        self.load_next_pair()

    def delete_image(self, side):
        path = self.img1_path if side == 'left' else self.img2_path

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {os.path.basename(path)}?"):
            try:
                if os.path.exists(path):
                    os.remove(path)
                if path in self.domain_map:
                    del self.domain_map[path]
                    save_domain_map(self.domain_map)

                # Reload: The current pair is now invalid (one file missing), so load_next_pair will auto-skip it
                self.load_next_pair()
            except Exception as e:
                messagebox.showerror("Error", str(e))



# --- Main Execution Block ---
if __name__ == "__main__":
    for path in [SOURCE_DIR, PROCESSED_DIR]:
        if not os.path.exists(path): os.makedirs(path)
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f: f.write('image1_path,image2_path,domain1,domain2,label\n')

    # SVG Pre-process
    svg_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.svg')]
    for s in svg_files:
        sp = os.path.join(SOURCE_DIR, s); pp = os.path.join(SOURCE_DIR, os.path.splitext(s)[0] + '.png')
        if not os.path.exists(pp):
            try: cairosvg.svg2png(url=sp, write_to=pp); os.remove(sp)
            except: pass

    # Phase 1
    d_root = tk.Tk(); DomainClassificationApp(d_root); d_root.mainloop()

    # Clash Resolution
    unlabeled = set(f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(VALID_EXTENSIONS))
    processed = set(f for f in os.listdir(PROCESSED_DIR) if f.lower().endswith(VALID_EXTENSIONS))
    clashes = list(unlabeled.intersection(processed))
    if clashes:
        r_root = tk.Tk(); DuplicateResolverApp(r_root, clashes); r_root.mainloop()

    # Phase 2
    m_root = tk.Tk(); ImageLabeler(m_root); m_root.mainloop()
