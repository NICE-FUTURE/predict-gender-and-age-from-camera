# 【Demo】基于CNN实现对摄像头捕捉人脸的性别和年龄预测 【效果不好，当时仅作为思路验证，后续未做更新】

### 如何使用

- 将data.zip解压到data目录下（data.zip更新为RGB图像，体积较大），训练集所在路径应为 `./data/trainset/`，测试集所在路径应为 `./data/testset/`
- 安装所需的第三方库 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` 
- 修改两处路径，将其指向OpenCV环境中对应的xml文件。分别是 `run.py`第11行 和 `data/process_wiki_data.py`第12行
- 训练年龄预测模型 `python train4age.py`
- 训练性别分类模型 `train4gender.py`
- 开始使用 `python run.py`

### 主要组成部分

- 人脸识别模块（使用OpenCV自带的人脸识别功能，效果一般）
- 性别分类模型（使用keras构建的一个CNN）
- 年龄预测模型（使用keras构建的一个CNN）

### 实现思路

![demo](https://img.hxhen.com/20200414231955.png)

### 准备数据

原始数据来源于 [https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar)
原始数据集包含的图片数量很多，我从中筛选了大约10000张图片（筛选条件为：由OpenCV识别出的face数目为1、性别已知、男女各约5000张）
将图片由RGB图转换为灰度图，将图片尺寸压缩为20*20
从10000张图片中抽取1000张（男女比例相当）作为测试集，其余作为训练集

### 训练模型

性别分类模型和年龄预测模型的结构相同，均为两层卷积层，一层池化层，一层全连接层，一层输出层
性别分类模型的输出层使用sigmoid激活，损失函数选用binary_crossentropy
年龄预测模型的输出层使用relu激活，损失函数选用mean_absolute_error

性别分类：
```python
# train4gender.py
def train_CNN(x_train, y_train, x_test, y_test):
    input_layer = Input(shape=(20, 20, 1), name="input_layer")

    conv2d_layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2d_layer = Conv2D(64, (3, 3), activation='relu')(conv2d_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv2d_layer)
    pool_layer = Dropout(0.25)(pool_layer)
    flatten_layer = Flatten()(pool_layer)
    hidden_layer = Dense(128, activation='relu')(flatten_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    output_layer = Dense(units=1, activation="sigmoid", name="output_layer")(hidden_layer)
    model = Model(input_layer, output_layer)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=2)
    return model, history
```

年龄预测：
```python
# train4age.py
def train_CNN(x_train, y_train, x_test, y_test):
    input_layer = Input(shape=(20, 20, 1), name="input_layer")

    conv2d_layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2d_layer = Conv2D(64, (3, 3), activation='relu')(conv2d_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv2d_layer)
    pool_layer = Dropout(0.25)(pool_layer)
    flatten_layer = Flatten()(pool_layer)
    hidden_layer = Dense(128, activation='relu')(flatten_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    output_layer = Dense(units=1, activation="relu", name="output_layer")(hidden_layer)
    model = Model(input_layer, output_layer)
    model.compile(optimizer="adam", loss="mae")
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=2)
    return model, history
```

### 模型效果

模型效果并不好，一是模型构建的简陋，二是数据量不大。模型方面有很多CNN模型如：AlexNet、VGG、Inception、ResNet等都可以进一步了解。

个人推荐这篇文章：[https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-cnn%E6%BC%94%E5%8C%96%E5%8F%B2-alexnet-vgg-inception-resnet-keras-coding-668f74879306](https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-cnn%E6%BC%94%E5%8C%96%E5%8F%B2-alexnet-vgg-inception-resnet-keras-coding-668f74879306)

这是一次训练过程的记录

train4gender_history.png

![train4gender_history.png](https://img.hxhen.com/20200414234421.png)

train4age_history.png

![train4age_history.png](https://img.hxhen.com/20200414234451.png)

能够实时对摄像头拍摄的图像处理并展示。
下面是一张不敢恭维的效果图，结果有点惨不忍睹……（组合的四张图片挑选自imdb-wiki数据集）

![](https://img.hxhen.com/20200415150416.png)

![](https://img.hxhen.com/20200415150435.png)

### 完整代码 
[https://github.com/nice-future/predict-gender-and-age-from-camera](https://github.com/nice-future/predict-gender-and-age-from-camera)
