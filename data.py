import numpy as np
from skimage import io
import os
import math
from keras.utils import Sequence

from config import Config

class Dataset(Sequence):

    def __init__(self, file_dir, load_type):
        self.file_dir = file_dir
        self.load_type = load_type  # load_type: age/gender
        self.filenames = os.listdir(file_dir)
        if self.load_type == "age":
            self.batch_size = Config.age_batch_size
        elif self.load_type == "gender":
            self.batch_size = Config.gender_batch_size

    def __len__(self):
        return math.ceil(len(self.filenames)/self.batch_size) 

    def __getitem__(self, idx):
        batch_filenames = self.filenames[idx*self.batch_size: (idx+1)*self.batch_size]
        images, ages, genders = self._load_images(batch_filenames)
        images = images/255
        # ages = (ages-Config.min_age)/(Config.max_age-Config.min_age)
        if self.load_type == "age":
            return (images, ages)
        elif self.load_type == "gender":
            return (images, genders)

    def _load_images(self, batch_filenames):
        dataset = None
        ages = []
        genders = []
        for filename in batch_filenames:
            # 图片特征
            ages.append(int(filename.split(".")[0].split("-")[1]))
            genders.append(int(filename.split(".")[0].split("-")[2]))
            # 图片数据
            image_array = io.imread(os.path.join(self.file_dir, filename))
            image_array = np.expand_dims(image_array, axis=0)
            if dataset is None:
                dataset = image_array
            else:
                dataset = np.append(dataset, image_array, axis=0)
        return dataset, np.array(ages), np.array(genders)
