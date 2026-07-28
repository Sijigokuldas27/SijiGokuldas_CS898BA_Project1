import os
import shutil
import random

# Where the correct fish dataset lives locally
source_base = r"C:\Users\sijip\Downloads\Fish\Fish"

# Where we'll copy our organized subset
dest_base = "fish_data"

# Split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Get list of species folder names
species_list = [
    name for name in os.listdir(source_base)
    if os.path.isdir(os.path.join(source_base, name))
]

print("Found species:", species_list)

# Set a fixed random seed so results are reproducible
random.seed(42)

for species in species_list:
    images_folder = os.path.join(source_base, species)

    all_images = [
        f for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(all_images)

    # Use all available images (dataset is small, no cap needed)
    selected_images = all_images

    # Calculate split sizes
    n_total = len(selected_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_images = selected_images[:n_train]
    val_images = selected_images[n_train:n_train + n_val]
    test_images = selected_images[n_train + n_val:]

    # Copy images into train/val/test folders
    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name, image_list in splits.items():
        dest_folder = os.path.join(dest_base, split_name, species)
        os.makedirs(dest_folder, exist_ok=True)

        for img_name in image_list:
            src_path = os.path.join(images_folder, img_name)
            dst_path = os.path.join(dest_folder, img_name)
            shutil.copy(src_path, dst_path)

    print(f"{species}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

print("\nDone! Organized dataset is in the 'fish_data' folder.")