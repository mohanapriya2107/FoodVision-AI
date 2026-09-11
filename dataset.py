import os
import shutil
import random

# ==========================================
# Source folder
# ==========================================

source_dir = r"Food Classification dataset"

# ==========================================
# Destination folder
# ==========================================

output_dir = r"food"

# ==========================================
# Number of images required
# ==========================================

train_count = 250
test_count = 20
valid_count = 50

total_required = train_count + test_count + valid_count

# ==========================================
# Supported image formats
# ==========================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# ==========================================
# Create main folders
# ==========================================

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
valid_dir = os.path.join(output_dir, "valid")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# ==========================================
# Iterate through every food class
# ==========================================

for folder_name in os.listdir(source_dir):

    folder_path = os.path.join(source_dir, folder_name)

    # Skip files
    if not os.path.isdir(folder_path):
        continue

    # ==========================================
    # Get all images
    # ==========================================

    images = [
        image
        for image in os.listdir(folder_path)
        if image.lower().endswith(image_extensions)
    ]

    # ==========================================
    # Check number of images
    # ==========================================

    if len(images) < total_required:

        print(
            f"Skipping {folder_name}: "
            f"Only {len(images)} images available, "
            f"but {total_required} required."
        )

        continue

    # ==========================================
    # Shuffle images
    # ==========================================

    random.shuffle(images)

    # ==========================================
    # Select images
    # ==========================================

    train_images = images[:train_count]

    test_images = images[
        train_count:
        train_count + test_count
    ]

    valid_images = images[
        train_count + test_count:
        train_count + test_count + valid_count
    ]

    # ==========================================
    # Create class folders
    # ==========================================

    train_class_dir = os.path.join(
        train_dir,
        folder_name
    )

    test_class_dir = os.path.join(
        test_dir,
        folder_name
    )

    valid_class_dir = os.path.join(
        valid_dir,
        folder_name
    )

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)
    os.makedirs(valid_class_dir, exist_ok=True)

    # ==========================================
    # Copy training images
    # ==========================================

    for image in train_images:

        source_image = os.path.join(
            folder_path,
            image
        )

        destination_image = os.path.join(
            train_class_dir,
            image
        )

        shutil.copy2(
            source_image,
            destination_image
        )

    # ==========================================
    # Copy testing images
    # ==========================================

    for image in test_images:

        source_image = os.path.join(
            folder_path,
            image
        )

        destination_image = os.path.join(
            test_class_dir,
            image
        )

        shutil.copy2(
            source_image,
            destination_image
        )

    # ==========================================
    # Copy validation images
    # ==========================================

    for image in valid_images:

        source_image = os.path.join(
            folder_path,
            image
        )

        destination_image = os.path.join(
            valid_class_dir,
            image
        )

        shutil.copy2(
            source_image,
            destination_image
        )

    # ==========================================
    # Display result
    # ==========================================

    print(
        f"{folder_name}: "
        f"Train={len(train_images)}, "
        f"Test={len(test_images)}, "
        f"Valid={len(valid_images)}"
    )


print("\nCompleted!")