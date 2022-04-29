# 【Demo】基于CNN实现对人脸的性别和年龄预测（2022.4.7 更新）

### 更新说明

1. 深度学习框架由Keras替换为PyTorch
2. 模型结构由少量的卷积层替换为成熟的ResNet作为主干，并将年龄预测和性别预测统一到一个模型中，采用双分支输出结构。
3. 目前在验证集的性别预测准确率为 91%
4. 实际体验差强人意，但与去年5月份的试水版本相比，效果会更好
5. ~~【2022.4.22】模型权重已上传 <a href="#">click to download</a> ，玩得愉快\~~~
6. 【2022.4.29】删除了代码仓库中的权重文件，如有需要，欢迎移步 <a href="https://github.com/NICE-FUTURE/predict-gender-and-age-from-camera/releases/">Release</a> 页面下载~

### 如何使用

- 将data.zip解压到data目录下（data.zip更新为RGB图像，体积较大），训练集所在路径应为 `./data/trainset/`，测试集所在路径应为 `./data/testset/`
- 安装所需的第三方库 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` 
- 修改两处路径，将其指向OpenCV环境中对应的xml文件。分别是 `run.py`第46行 和 `data/process_wiki_data.py`第12行
- 使用GPU训练模型 `./scripts/run_gpu.ps1`
- 用视频测试模型 `python .\run.py --pretrain_weight_path .\middle\models\test-best.pth --mode video`

### 实现思路

![structure](./samples/structure.png)

### 数据处理

- 原始数据来源于 [https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar)
- 原始数据集包含的图片数量很多，我从中筛选了大约10000张图片（筛选条件为：由OpenCV识别出的face数目为1、性别已知、男女各约5000张）
- 图片尺寸统一为 100x100，文件名格式统一为 `编号-年龄-性别.png`，其中性别1代表男性，0代表女性
- 从10000张图片中抽取约1000张（男女比例相当）作为测试集，其余作为训练集

### 模型结构

- 性别预测分支和年龄预测分支共用ResNet50主干，年龄预测分支和性别预测分支各包含三层卷积层
- 性别预测分支使用交叉熵损失函数
- 年龄预测分支使用均方差损失函数

### 模型效果

训练过程的记录

![history.png](./middle/history/test-2022-04-07%2011.35.16.210418.png)

这是对一张组合图像的处理结果（组合的四张图片选自imdb-wiki数据集的原始图像）

![sample_result.png](samples/sample_result.png)

<a id="sample.mp4" href="./samples/sample.mp4">测试用的视频</a>剪辑自：[https://www.di.ens.fr/~laptev/download/drinking.avi](https://www.di.ens.fr/~laptev/download/drinking.avi)

