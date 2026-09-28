from pathlib import Path

from datasets import load_dataset
from PIL import Image


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
TOMATO_DIR = RAW_DIR / "tomato"

TOMATO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Tomato classes
# ============================================================

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two_spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]


# ============================================================
# Create class folders
# ============================================================

class_to_folder = {
    class_name: TOMATO_DIR / class_name
    for class_name in TOMATO_CLASSES
}

for folder in class_to_folder.values():
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load PlantVillage from Hugging Face
# ============================================================

print("=" * 70)
print("Loading PlantVillage dataset from Hugging Face")
print("=" * 70)

dataset = load_dataset(
    "mohanty/PlantVillage",
    name="default",
)

print()
print("Dataset loaded successfully.")
print(dataset)
print()


# ============================================================
# Inspect available splits
# ============================================================

print("=" * 70)
print("Available dataset splits")
print("=" * 70)

for split_name, split_data in dataset.items():
    print(f"{split_name}: {len(split_data)} images")

print()


# ============================================================
# Select available split
# ============================================================

if "train" in dataset:
    data = dataset["train"]
elif "test" in dataset:
    data = dataset["test"]
else:
    first_split = list(dataset.keys())[0]
    data = dataset[first_split]

print(f"Using split: {data}")
print()


# ============================================================
# Inspect label information
# ============================================================

print("=" * 70)
print("Dataset features")
print("=" * 70)

print(data.features)
print()


# ============================================================
# Determine label column
# ============================================================

label_column = None

for column_name in data.column_names:

    feature = data.features[column_name]

    if hasattr(feature, "names"):
        label_column = column_name
        break

if label_column is None:
    raise RuntimeError(
        "Could not automatically find the label column."
    )

print(f"Label column: {label_column}")
print()

label_names = data.features[label_column].names

print("Total classes:", len(label_names))
print()


# ============================================================
# Find tomato class IDs
# ============================================================

tomato_label_ids = {}

for class_name in TOMATO_CLASSES:

    if class_name in label_names:

        label_id = label_names.index(class_name)

        tomato_label_ids[label_id] = class_name

print("=" * 70)
print("Tomato classes found")
print("=" * 70)

for label_id, class_name in tomato_label_ids.items():
    print(f"{label_id}: {class_name}")

print()


if len(tomato_label_ids) != len(TOMATO_CLASSES):

    missing_classes = [
        class_name
        for class_name in TOMATO_CLASSES
        if class_name not in label_names
    ]

    raise RuntimeError(
        f"Missing tomato classes: {missing_classes}"
    )


# ============================================================
# Determine image column
# ============================================================

image_column = None

for column_name in data.column_names:

    if column_name == "image":
        image_column = column_name
        break

if image_column is None:

    raise RuntimeError(
        f"Could not find image column. "
        f"Available columns: {data.column_names}"
    )

print(f"Image column: {image_column}")
print()


# ============================================================
# Extract tomato images
# ============================================================

class_counts = {
    class_name: 0
    for class_name in TOMATO_CLASSES
}


print("=" * 70)
print("Extracting tomato images")
print("=" * 70)


for index, example in enumerate(data):

    label_id = int(example[label_column])

    # Skip non-tomato classes
    if label_id not in tomato_label_ids:
        continue

    class_name = tomato_label_ids[label_id]

    output_folder = class_to_folder[class_name]

    image_number = class_counts[class_name] + 1

    output_file = (
        output_folder
        / f"{class_name}_{image_number:05d}.jpg"
    )

    image = example[image_column]

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    image.save(
        output_file,
        format="JPEG",
        quality=95,
    )

    class_counts[class_name] += 1

    total_extracted = sum(class_counts.values())

    if total_extracted % 1000 == 0:

        print(
            f"Extracted {total_extracted} tomato images..."
        )


# ============================================================
# Final summary
# ============================================================

total_images = sum(class_counts.values())


print()
print("=" * 70)
print("DATASET ACQUISITION COMPLETE")
print("=" * 70)

for class_name, count in class_counts.items():

    print(
        f"{class_name:<55} {count:>5}"
    )

print("-" * 70)

print(
    f"{'Total tomato images':<55} "
    f"{total_images:>5}"
)

print()
print("Dataset location:")
print(TOMATO_DIR)

print()
print("Stage 2 dataset extraction completed successfully.")
