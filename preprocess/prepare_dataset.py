import os
import random
from datetime import datetime, timedelta

import numpy as np
import scipy.io as sio
from PIL import Image


def generate_all_txt(root, txt_dir, num_samples=None):
    """ 筛选符合要求的数据并生成 all.txt 文件
    """
    # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    data = sio.loadmat(os.path.join(root, "wiki.mat"))["wiki"][0][0]

    birth_times = data[0][0]  # 出生时间 matlab time format
    shot_times = data[1][0]  # 拍摄时间
    file_paths = data[2][0]  # 文件路径
    gender_flags = data[3][0]  # 0-female, 1-male, NaN-unknown
    # face_locations = data[5][0]  # wiki_crop.tar 是已经裁剪过的数据 四周的裁剪位置均相对于原图尺寸向外移动了40%
    face_scores = np.isinf(data[6][0])  # Inf means no face in the image, and returns the entire image
    second_face_scores = np.isnan(data[7][0])  # NaN means no second face in the image

    total = len(shot_times)
    cnt_valid = 0  # 实际遍历数目
    if num_samples is None:
        num_samples = total

    os.makedirs(txt_dir, exist_ok=True)

    f = open(os.path.join(txt_dir, "all.txt"), "w", encoding="utf-8")

    for i in range(total):
        # 获得图片对应的特征信息
        birth_time = int(birth_times[i])
        birth_time = (datetime.fromordinal(birth_time - 366) + timedelta(days=birth_time%1)).year
        # birth_time = int(file_path.split("_")[-2].split("-")[0])  # 从文件名获取出生年份
        age = int(shot_times[i]) - birth_time
        file_path = file_paths[i][0]
        gender_flag = gender_flags[i]
        has_first_face = not face_scores[i]
        has_second_face = not second_face_scores[i]

        if (
            has_first_face
            and not has_second_face
            and not np.isnan(gender_flag)
        ):
            gender_flag = int(gender_flag)
            f.write(f"{file_path},{age},{gender_flag}\n")  # file_path, age, gender
            cnt_valid += 1
            if cnt_valid >= num_samples:
                break

    print("cur:{}, valid:{}, total:{}".format(i+1, cnt_valid, total))


def count_male_female(txt_path):
    '''
    统计图片的男女比例
    '''
    male = 0
    female = 0
    total = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            file_path, age, gender = line.strip().split(",")
            if gender == "0":
                female += 1
            elif gender == "1":
                male += 1
            else:
                raise NotImplementedError
            total += 1

    print(f"male:{male}, female:{female}, total:{total}")


def split_train_test(txt_dir, ratio=0.6, seed=123):
    '''拆分训练集和测试集
    '''
    with open(os.path.join(txt_dir, "all.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)

    random.seed(seed)
    random.shuffle(lines)
    n_train = int(total * ratio)
    train_lines = lines[:n_train]
    val_lines = lines[n_train:]

    with open(os.path.join(txt_dir, "train.txt"), "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)
    with open(os.path.join(txt_dir, "val.txt"), "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line)


if __name__ == "__main__":
    pass
    # generate_all_txt(root="D:/dataset/wiki_crop", txt_dir="../data/wiki/")
    # split_train_test("../data/wiki/")
    # count_male_female("../data/wiki/all.txt")
    # count_male_female("../data/wiki/train.txt")
    # count_male_female("../data/wiki/val.txt")
