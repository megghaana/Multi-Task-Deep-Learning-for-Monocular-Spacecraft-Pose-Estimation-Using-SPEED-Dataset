import torch
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import SpeedDataset
from model import PoseCNN


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 2. Dataset paths
# --------------------------------------------------

DATASET_PATH = Path("speed/speed")

IMAGE_PATH = DATASET_PATH / "images" / "train"

TRAIN_JSON = DATASET_PATH / "train.json"


# --------------------------------------------------
# 3. Image transformation
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --------------------------------------------------
# 4. Load dataset
# --------------------------------------------------

dataset = SpeedDataset(
    TRAIN_JSON,
    IMAGE_PATH,
    transform
)

print("Total samples:", len(dataset))


# --------------------------------------------------
# 5. DataLoader
# --------------------------------------------------

test_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 6. Create model
# --------------------------------------------------

model = PoseCNN().to(device)


# --------------------------------------------------
# 7. Load trained weights
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "pose_cnn_baseline.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully!")


# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------

translation_errors = []
rotation_errors = []

mse_losses = []

criterion = torch.nn.MSELoss()


with torch.no_grad():

    for images, poses in test_loader:

        images = images.to(device)
        poses = poses.to(device)

        # Model prediction
        predictions = model(images)

        # MSE
        loss = criterion(predictions, poses)
        mse_losses.append(loss.item())

        # Move to CPU
        predictions = predictions.cpu().numpy()
        poses = poses.cpu().numpy()


        # ------------------------------------------
        # Calculate errors for every image
        # ------------------------------------------

        for pred, true in zip(predictions, poses):

            # --------------------------------------
            # Position
            # --------------------------------------

            predicted_position = pred[:3]
            true_position = true[:3]

            position_error = np.linalg.norm(
                predicted_position - true_position
            )

            translation_errors.append(position_error)


            # --------------------------------------
            # Quaternion
            # --------------------------------------

            predicted_quaternion = pred[3:]
            true_quaternion = true[3:]


            # Normalize quaternions
            predicted_quaternion = (
                predicted_quaternion /
                np.linalg.norm(predicted_quaternion)
            )

            true_quaternion = (
                true_quaternion /
                np.linalg.norm(true_quaternion)
            )


            # Quaternion dot product
            dot = np.abs(
                np.dot(
                    predicted_quaternion,
                    true_quaternion
                )
            )

            # Numerical safety
            dot = np.clip(dot, 0.0, 1.0)


            # Angular difference
            angle = 2 * np.arccos(dot)

            angle_degrees = np.degrees(angle)

            rotation_errors.append(angle_degrees)


# --------------------------------------------------
# 9. Results
# --------------------------------------------------

mean_translation_error = np.mean(
    translation_errors
)

mean_rotation_error = np.mean(
    rotation_errors
)

mean_mse = np.mean(
    mse_losses
)


print("\n")
print("=" * 50)
print("         POSE ESTIMATION RESULTS")
print("=" * 50)

print(
    f"Mean MSE Loss:          {mean_mse:.6f}"
)

print(
    f"Mean Translation Error: {mean_translation_error:.4f} meters"
)

print(
    f"Mean Rotation Error:    {mean_rotation_error:.4f} degrees"
)

print("=" * 50)