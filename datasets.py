from torch.utils.data import Dataset
from PIL import  Image
import os
import numpy as np


class BatchDataset(Dataset):

    def __init__(self, root, txt_dir, name, transform):
        self.root = root
        self.transform = transform
        with open(os.path.join(txt_dir, f"{name}.txt"), "r", encoding="utf-8") as f:
            self.lines = f.readlines()

    def __getitem__(self, idx):
        filename, age, gender = self.lines[idx].strip().split(",")
        image_path = os.path.join(self.root, filename)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        age = np.float32(int(age)/ 100.0) 
        gender = int(gender)

        return image, age, gender, filename

    def __len__(self):
        return len(self.lines)