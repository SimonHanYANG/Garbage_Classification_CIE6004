import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class GarbageDataset(Dataset):
    def __init__(self, filename, transform=None):
        self.data = open(filename, 'r').readlines()
        self.transform = transform
        self.label_to_int = {
            "battery": 0,
            "biological": 1,
            "brown-glass": 2,
            "cardboard": 3,
            "clothes": 4,
            "green-glass": 5,
            "metal": 6,
            "paper": 7,
            "plastic": 8,
            "shoes": 9,
            "trash": 10,
            "white-glass": 11,
            # Add more if there are more classes
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        path = line.strip()
        label = path.split('/')[1]  # Assumes format is "garbage_classification/battery/battery198.jpg"
        label_int = self.label_to_int[label]

        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label_int