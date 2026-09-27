import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

DATASET_PATH = Path("speed/speed")
IMAGE_PATH = DATASET_PATH / "images/train"
TRAIN_JSON = DATASET_PATH / "train.json"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class SpeedDataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform 
        with open(json_file, "r") as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image_name = sample["filename"]
        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("L") # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        position = sample["r_Vo2To_vbs_true"]
        quaternion = sample["q_vbs2tango"]
        pose = position + quaternion  # Concatenate position and quaternion
        pose = torch.tensor(pose, dtype=torch.float32)
        return image, pose
dataset = SpeedDataset(TRAIN_JSON, IMAGE_PATH, transform)
print("Number of samples:", len(dataset))


# -----------------------------
# Test one sample
# -----------------------------

image, pose = dataset[0]

print("Image tensor shape:", image.shape)
print("Pose:", pose)
print("Pose shape:", pose.shape)


# -----------------------------
# DataLoader
# -----------------------------

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# Test one batch

images, poses = next(iter(loader))

print("Batch image shape:", images.shape)
print("Batch pose shape:", poses.shape)