import numpy as np
from PIL import Image
import os

class Data(object):

    def __init__(self, trainpath="./data/trainset-s20/", testpath="./data/testset-s20/"):
        self.trainpath = trainpath
        self.testpath = testpath

    def _load_images(self, path):
        dataset = None
        ages = []
        genders = []
        for filename in os.listdir(path):
            # 图片特征
            ages.append(int(filename.split(".")[0].split("-")[1]))
            genders.append(int(filename.split(".")[0].split("-")[2]))
            # 图片数据
            image = Image.open(path+filename)
            if dataset is None:
                dataset = np.expand_dims(np.array(image).flatten(), axis=0)
            else:
                dataset = np.append(dataset, np.expand_dims(np.array(image).flatten(), axis=0), axis=0)
            image.close()
        return dataset, ages, genders

    def load4age(self):
        trainset, train_ages, _ = self._load_images(self.trainpath)
        testset, test_ages, _ = self._load_images(self.testpath)
        # minimun = min(testset)
        # maximun = max(testset)
        # testset = (testset-minimun)/(maximun-minimun)
        # trainset = trainset/100
        # testset = testset/100
        return trainset, train_ages, testset, test_ages

    def load4gender(self):
        trainset, _, train_genders = self._load_images(self.trainpath)
        testset, _, test_genders = self._load_images(self.testpath)
        trainset = trainset/255
        testset = testset/255
        return trainset, train_genders, testset, test_genders

if __name__ == "__main__":
    data = Data()
    x_train, y_train, x_test, y_test = data.load4gender()
    print("x_train:{}, y_train:{}\nx_test:{}, y_test:{}".format(x_train.shape, len(y_train), x_test.shape, len(y_test)))
    print(x_train[:5])
    print(y_train[:5])
