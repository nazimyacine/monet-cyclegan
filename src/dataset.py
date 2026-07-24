import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

class ArtDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        self.images_A = [f for f in os.listdir(root_A) if f.endswith((".jpg", ".png", ".jpeg"))]
        self.images_B = [f for f in os.listdir(root_B) if f.endswith((".jpg", ".png", ".jpeg"))]

        self.len_A = len(self.images_A)
        self.len_B = len(self.images_B)

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, index):
        img_A = self.images_A[index % self.len_A]
        img_B = self.images_B[index % self.len_B]

        path_A = os.path.join(self.root_A, img_A)
        path_B = os.path.join(self.root_B, img_B)

        img_A = np.array(Image.open(path_A).convert("RGB"))
        img_B = np.array(Image.open(path_B).convert("RGB"))

        if self.transform:
            aug = self.transform(image=img_A, image0=img_B)
            img_A = aug["image"]
            img_B = aug["image0"]

        return img_A, img_B