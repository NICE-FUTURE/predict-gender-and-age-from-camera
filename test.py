import cv2
import numpy as np
from keras.models import load_model

import os

from config import Config

gender_model = load_model("cnn4gender_best.h5")
age_model = load_model("cnn4age_best.h5")

def test():

    root = "../data/testset/"
    filenames = os.listdir(root)

    with open("testset_result.csv", "w", encoding="utf-8") as f:
        f.write("id,predict gender,ground truth gender,predict age,ground truth age\n")
        total = len(filenames)
        cnt = 0
        for filename in filenames:
            cnt += 1
            print("{}/{}".format(cnt, total), end="\r")
            path = os.path.join(root, filename)
            image = cv2.imread(path)
            p_gender = get_gender(image)
            p_age = get_age(image)
            # p_age = p_age*(Config.max_age-Config.min_age) + Config.min_age
            key, g_age, g_gender = filename.split(".")[0].split("-")
            g_gender = "female" if g_gender == "0" else "male"
            f.write("{},{},{},{},{}\n".format(key,p_gender,g_gender,p_age,g_age))
        print("{}/{}".format(cnt, total))


def get_gender(image):
    image = np.expand_dims(image, axis=0)
    gender = gender_model.predict(image)
    if gender[0] == 1:
        return 'male'
    else:
        return 'female'


def get_age(image):
    image = np.expand_dims(image, axis=0)
    age = age_model.predict(image)
    age = round(age[0][0])
    return age


if __name__ == "__main__":
    test()
