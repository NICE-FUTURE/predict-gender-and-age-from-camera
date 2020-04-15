import os
import random
from PIL import Image
from collections import Counter

def statistic_images():
    '''
    统计图片size
    '''
    trainset = os.listdir("./trainset/")
    result = []
    for filename in trainset:
        image = Image.open("./trainset/"+filename)
        result.append(image.size)
        image.close()
    x = [item[0] for item in result]
    y = [item[1] for item in result]
    print("均值：", (sum(x)/len(x), sum(y)/len(y)) )
    print("众数：", list(Counter(result).items())[0])

def resize_images(paths=[]):
    '''
    更改图片尺寸
    '''

    for directory in paths:
        if not os.path.exists(directory+"-s20/"):
            os.mkdir(directory+"-s20/")

        dataset = os.listdir(directory)
        for filename in dataset:
            image = Image.open(directory+"/"+filename)
            try:
                # image.resize((20, 20)).convert("L").save(directory+"-s20/"+filename[:-3]+"png")
                image.resize((20, 20)).save(directory+"-s20/"+filename[:-3]+"png")
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

def split_train_test(root):
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
        if cnt < 1000 and gender == "0" and flag == "0":
            testset.append(filename)
            flag = "1"
            cnt += 1
        elif cnt < 1000 and gender == "1" and flag == "1":
            testset.append(filename)
            flag = "0"
            cnt += 1
        else:
            trainset.append(filename)

    for name in trainset:
        os.rename(root+name, "./trainset/"+name)
    for name in testset:
        os.rename(root+name, "./testset/"+name)

# rename_images(['images-10000'])
# count_male_female('images-10000')  # 确定性别分类时判定为正的阈值
# resize_images(["images-10000"])
split_train_test("images-10000-s20/")
