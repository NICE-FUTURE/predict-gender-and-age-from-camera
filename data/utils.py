import os
import shutil
import random
from PIL import Image
from collections import Counter

def statistic_images(path):
    '''
    统计图片size
    '''
    trainset = os.listdir(path)
    result = []
    for filename in trainset:
        image = Image.open(path+"/"+filename)
        result.append(image.size)
        image.close()
    x = [item[0] for item in result]
    y = [item[1] for item in result]
    print("均值：", (sum(x)/len(x), sum(y)/len(y)) )
    print("众数：", list(Counter(result).items())[0])

def resize_images(paths=[], size=20):
    '''
    更改图片尺寸
    '''
    for directory in paths:
        if not os.path.exists(directory+"-s{}/".format(size)):
            os.mkdir(directory+"-s{}/".format(size))

        dataset = os.listdir(directory)
        for filename in dataset:
            image = Image.open(directory+"/"+filename)
            try:
                image.resize((size, size)).save(directory+"-s"+str(size)+"/"+filename[:-3]+"png")
            except:
                pass
            image.close()

def rename_images(paths=[]):
    '''
    保证图片序号唯一
    '''
    roots = paths
    root = "./images-renamed/"

    cur = 0
    for root1 in roots:
        filenames = os.listdir(root1)
        cnt = 0
        total = len(filenames)
        for filename in filenames:
            cnt += 1
            cur += 1
            print("{}/{}/{}".format(cnt, total, cur), end="\r")
            [header, body1, body2] = filename.split("-")
            os.rename(root1+filename, root+"{}-{}-{}".format(cur, body1, body2))
    print()

def count_male_female(path):
    '''
    统计图片的男女比例
    '''
    male = 0
    female = 0
    filenames = os.listdir(path)
    print("total:{}".format(len(filenames)))
    for filename in filenames:
        gender = filename.split("-")[2][0]
        if gender == "0":
            female += 1
        elif gender == "1":
            male += 1

    print("male:{}, female:{}".format(male, female))

def split_train_test(root, n_test):
    '''
    拆分训练集和测试集
    '''

    if not os.path.exists("./trainset/"):
        os.mkdir("./trainset/")
    if not os.path.exists("./testset/"):
        os.mkdir("./testset/")

    filenames = os.listdir(root)
    random.shuffle(filenames)

    trainset = []
    testset = []
    flag = "0"
    cnt = 0
    for filename in filenames:
        gender = filename[:-4].split("-")[-1]
        if cnt < n_test and gender == "0" and flag == "0":
            testset.append(filename)
            flag = "1"
            cnt += 1
        elif cnt < n_test and gender == "1" and flag == "1":
            testset.append(filename)
            flag = "0"
            cnt += 1
        else:
            trainset.append(filename)

    for name in trainset:
        os.rename(root+name, "./trainset/"+name)
    for name in testset:
        os.rename(root+name, "./testset/"+name)
    os.rmdir(root)


def resplit_images():
    train_filenames = os.listdir("./trainset-9000/")
    test_filenames = os.listdir("./testset-1000/")

    female_cnt = 0
    male_cnt = 0
    for filename in train_filenames:
        gender = filename[:-4].split("-")[-1]
        if gender == "0" and female_cnt < 500:
            shutil.copyfile("./trainset-9000/"+filename, "./trainset/"+filename)
            female_cnt += 1
        elif gender == "1" and male_cnt < 500:
            shutil.copyfile("./trainset-9000/"+filename, "./trainset/"+filename)
            male_cnt += 1

    female_cnt = 0
    male_cnt = 0
    for filename in test_filenames:
        gender = filename[:-4].split("-")[-1]
        if gender == "0" and female_cnt < 50:
            shutil.copyfile("./testset-1000/"+filename, "./testset/"+filename)
            female_cnt += 1
        elif gender == "1" and male_cnt < 50:
            shutil.copyfile("./testset-1000/"+filename, "./testset/"+filename)
            male_cnt += 1


resize_images(["images-1100"], 100)
split_train_test("images-1100-s100/", 100)

# count_male_female('images-1100')  # 确定性别分类时判定为正的阈值
# statistic_images("images-1100")  # 统计图片size
# rename_images(['images-1100'])  # 保证图片序号唯一，由多次执行process导致
# resplit_images()  # 从大数据集分割小数据集（临时用）
