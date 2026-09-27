import json
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = Path("speed/speed")

IMAGE_PATH = DATASET_PATH / "images/train"
TRAIN_JSON = DATASET_PATH / "train.json"

with open(TRAIN_JSON, "r") as file:
    data = json.load(file)

print("Number of samples:", len(data))

sample = data[0]

filename = sample["filename"]
quaternion = sample["q_vbs2tango"]
position = sample["r_Vo2To_vbs_true"]

print("Filename:", filename)
print("Quaternion:", quaternion)
print("Position:", position)


image_path = IMAGE_PATH / filename

image = Image.open(image_path)

print("Image size:", image.size)
print("Image mode:", image.mode)

plt.imshow(image, cmap="gray")
plt.title(filename)
plt.axis("off")
plt.show()