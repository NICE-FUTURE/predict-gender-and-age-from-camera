from torch.utils.data import Dataset
from PIL import  Image
import os
import numpy as np


class BatchDataset(Dataset):

    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.filenames = os.listdir(image_dir)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        age = np.float32(filename.split(".")[0].split("-")[1])
        gender = int(filename.split(".")[0].split("-")[2])

        return image, age, gender, filename

    def __len__(self):
        return len(self.filenames)