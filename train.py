import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from data_loader import SpeedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

DATASET_PATH = Path("speed/speed")

IMAGE_PATH = DATASET_PATH / "images" / "train"
TRAIN_JSON = DATASET_PATH / "train.json"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = SpeedDataset(TRAIN_JSON, IMAGE_PATH, transform)
print("total sample:", len(dataset))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

class PoseCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.fc(x)

        return x


model = PoseCNN().to(device)

print(model)

position_criterion = nn.MSELoss()
quaternion_criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)



epochs = 20

for epoch in range(epochs):

    model.train()

    train_loss = 0.0

    for images, poses in train_loader:

        images = images.to(device)
        poses = poses.to(device)

        # Forward pass
        predictions = model(images)

        # Calculate loss
        position_loss = position_criterion(
            predictions[:, :3],
            poses[:, :3]
        )

        quaternion_loss = quaternion_criterion(
            predictions[:, 3:],
            poses[:, 3:]
        )

        loss = position_loss + quaternion_loss

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, poses in val_loader:

            images = images.to(device)
            poses = poses.to(device)

            predictions = model(images)

            position_loss = position_criterion(
                predictions[:, :3],
                poses[:, :3]
            )

            quaternion_loss = quaternion_criterion(
                predictions[:, 3:],
                poses[:, 3:]
            )

            loss = position_loss + quaternion_loss

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )



torch.save(
    model.state_dict(),
    "pose_cnn_baseline.pth"
)

print("Model saved!")
