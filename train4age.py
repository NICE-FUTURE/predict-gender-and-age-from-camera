from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

import time

from config import Config
from data import Dataset
from models import *


def visualize_history(history):

    plt.subplot("111")
    plt.plot(history["loss"], c="blue", label="loss")
    plt.plot(history["val_loss"], c="orange", label="val_loss")
    plt.legend()

    plt.savefig("train4age_history-{}.png".format(time.time()))

if __name__ == "__main__":
    trainset = Dataset(file_dir="../data/trainset/", load_type="age")
    testset = Dataset(file_dir="../data/testset/", load_type="age")

    # 训练模型
    # model = cnn_regression(Config.width, Config.height, Config.channel)
    model = vgg16_regression(Config.width, Config.height, Config.channel)
    adam = Adam(lr=Config.age_lr)
    model.compile(optimizer="adam", loss="mse")

    start_time = time.time()
    
    checkpoint = ModelCheckpoint("./cnn4age_best.h5", monitor='val_loss', verbose=1, 
            save_best_only=True, mode='auto', period=1)
    def scheduler(epoch, lr):
        # print("epoch:{}, lr:{}".format(epoch, 1e-7 * (2**epoch)))
        # return 1e-7 * (2**epoch)  # 寻找最合适的learning rate, 最佳 epoch 16, lr 0.0016384左右
        return lr
    lr_scheduler = LearningRateScheduler(scheduler)  # 适时调整learning rate
    history = model.fit_generator(generator=trainset, validation_data=testset, \
            epochs=Config.age_epochs, verbose=2, callbacks=[checkpoint, lr_scheduler])

    stop_time = time.time()
    print("training time:{}min".format((stop_time-start_time)/60))

    # 去除迭代过大的loss, 更好展示loss下降趋势
    history_loss = {}
    for name in ("loss", "val_loss"):
        loss = []
        minimum = min(history.history[name])
        for item in history.history[name]:
            if item > minimum*5000:
                loss.append(loss[-1])
            else:
                loss.append(item)
        history_loss[name] = loss

    visualize_history(history_loss)
