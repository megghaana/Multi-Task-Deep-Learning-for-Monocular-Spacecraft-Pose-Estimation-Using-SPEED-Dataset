import torch
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader, Subset
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
# 3. Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --------------------------------------------------
# 4. Load complete dataset
# --------------------------------------------------

dataset = SpeedDataset(
    TRAIN_JSON,
    IMAGE_PATH,
    transform
)

print("Total samples:", len(dataset))


# --------------------------------------------------
# 5. Load validation indices
# --------------------------------------------------

validation_indices = torch.load(
    "validation_indices.pth",
    weights_only=True
)

print(
    "Validation samples:",
    len(validation_indices)
)


# --------------------------------------------------
# 6. Create validation dataset
# --------------------------------------------------

validation_dataset = Subset(
    dataset,
    validation_indices
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 7. Create model
# --------------------------------------------------

model = PoseCNN().to(device)


# --------------------------------------------------
# 8. Load trained model
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
# 9. Evaluation
# --------------------------------------------------

translation_errors = []
rotation_errors = []
mse_losses = []

criterion = torch.nn.MSELoss()


with torch.no_grad():

    for batch_idx, (images, poses) in enumerate(validation_loader):

        images = images.to(device)
        poses = poses.to(device)

        predictions = model(images)

        # MSE
        loss = criterion(
            predictions,
            poses
        )

        mse_losses.append(
            loss.item()
        )

        # Move to CPU
        predictions = predictions.cpu().numpy()
        poses = poses.cpu().numpy()

        # Calculate errors
        for pred, true in zip(
            predictions,
            poses
        ):

            # ------------------------------
            # Position error
            # ------------------------------

            predicted_position = pred[:3]
            true_position = true[:3]

            position_error = np.linalg.norm(
                predicted_position - true_position
            )

            translation_errors.append(
                position_error
            )


            # ------------------------------
            # Quaternion error
            # ------------------------------

            predicted_quaternion = pred[3:]
            true_quaternion = true[3:]

            predicted_quaternion = (
                predicted_quaternion /
                np.linalg.norm(predicted_quaternion)
            )

            true_quaternion = (
                true_quaternion /
                np.linalg.norm(true_quaternion)
            )

            dot = np.abs(
                np.dot(
                    predicted_quaternion,
                    true_quaternion
                )
            )

            dot = np.clip(
                dot,
                0.0,
                1.0
            )

            angle = 2 * np.arccos(dot)

            angle_degrees = np.degrees(
                angle
            )

            rotation_errors.append(
                angle_degrees
            )


        # Progress
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Processed "
                f"{batch_idx + 1}/"
                f"{len(validation_loader)} batches"
            )


# --------------------------------------------------
# 10. Final results
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
print("=" * 55)
print("          VALIDATION RESULTS")
print("=" * 55)

print(
    f"Mean MSE Loss:          {mean_mse:.6f}"
)

print(
    f"Mean Translation Error: "
    f"{mean_translation_error:.4f} meters"
)

print(
    f"Mean Rotation Error:    "
    f"{mean_rotation_error:.4f} degrees"
)

print("=" * 55)