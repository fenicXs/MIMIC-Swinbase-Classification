from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MIMICCXRJPGDataset(Dataset):
    """MIMIC-CXR-JPG dataset from a prepared CSV.

    Expected CSV columns:
      - rel_path: relative path like files/p10/p10000032/s50414267/<dicom_id>.jpg
      - one column per class in class_names (0/1 float/int)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mimic_root: str | Path,
        class_names: List[str],
        image_size: int = 512,
        train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.mimic_root = Path(mimic_root)
        self.class_names = list(class_names)
        self.image_size = int(image_size)
        self.train = bool(train)

        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size + 64, self.image_size + 64)),
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = str(row["rel_path"])
        img_path = self.mimic_root / rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # MIMIC JPGs are often stored as grayscale; convert to RGB for ImageNet backbones
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        labels = torch.tensor(
            row[self.class_names].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        return image, labels
