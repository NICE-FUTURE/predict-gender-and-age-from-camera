'''
1. 从wiki_crop里面按一些条件筛选图片
2. 将人脸部分裁剪出来
3. 文件名包含性别和年龄
4. 放入images-10000文件夹
'''

import scipy.io as sio
import cv2
import os

face_cascade = cv2.CascadeClassifier('C:/Users/23755/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

root = "./wiki_crop/"
path = "wiki.mat"
data = sio.loadmat(root+path)["wiki"][0][0]

# birth_times = data[0][0]
shot_times = data[1][0]
file_paths = data[2][0]
gender_flags = data[3][0]  # 0-female, 1-male, NaN-unknown
# person_names = data[4][0]
face_locations = data[5][0]
face_scores = data[6][0]  # Inf means no face in the image, and returns the entire image
second_face_scores = data[7][0]  # NaN means no second face in the image

total = len(shot_times)
cnt = 0  # 有效样例数目
infact = 0  # 实际遍历数目
male = 0
female = 0
dataset_size = 10000

if not os.path.exists("./images-{}/".format(dataset_size)):
    os.mkdir("./images-{}/".format(dataset_size))

for i in range(total):
    
    # 显示进度
    cnt += 1
    infact += 1
    if cnt > dataset_size:
        break
    print("{}/{}/{}".format(cnt, infact, total), end="\r")

    # 获得图片对应的特征信息
    shot_time = int(shot_times[i])
    file_path = file_paths[i][0]
    gender_flag = gender_flags[i]
    face_score = face_scores[i]
    second_face_score = second_face_scores[i]
    birth_time = int(file_path.split("_")[-2].split("-")[0])  # 从文件名获取出生年份
    age = shot_time - birth_time  # 按周岁计

    # 只使用face数目为1，性别为0或1
    if face_score < 0 or second_face_score < 10 or str(gender_flag) == "nan" \
            or int(gender_flag) not in (0,1):
        cnt -= 1
        continue

    # 控制男女数目各5000
    gender_flag = int(gender_flag)
    if gender_flag == 0 and female >= dataset_size//2:
            cnt -= 1
            continue
    elif gender_flag == 1 and male >= dataset_size//2:
            cnt -= 1
            continue

    # 识别脸部位置
    try:
        img = cv2.imread(root+file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    except:
        cnt -= 1
        continue
    if len(faces) == 0:
        cnt -= 1
        continue

    # 累加男女人数
    gender_flag = int(gender_flag)
    if gender_flag == 0:
        female += 1
    elif gender_flag == 1:
        male += 1

    # 只保留脸部，保留灰度图
    x, y, w, h = faces[0]
    img = gray[x:x+w, y:y+h]

    cv2.imwrite("images-{}/{}-{}-{}.jpg".format(dataset_size, cnt, age, gender_flag), img)
