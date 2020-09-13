from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import time

from config import Config
from data import Dataset
from models import *


def visualize_history(history):

    plt.subplot("211")
    plt.plot(history["loss"], c="blue", label="loss")
    plt.plot(history["val_loss"], c="orange", label="val_loss")
    plt.legend()

    plt.subplot("212")
    plt.plot(history["acc"], c="blue", label="acc")
    plt.plot(history["val_acc"], c="orange", label="val_acc")
    plt.legend()

    plt.savefig("train4gender_history-{}.png".format(time.time()))

if __name__ == "__main__":

    trainset = Dataset(file_dir="../data/trainset/", load_type="gender")
    testset = Dataset(file_dir="../data/testset/", load_type="gender")

    # model = cnn_classification(Config.height, Config.width, Config.channel)
    model = vgg16_classification(Config.height, Config.width, Config.channel)
    adam = Adam(lr=Config.gender_lr)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

    start_time = time.time()
    
    checkpoint = ModelCheckpoint("./cnn4gender_best.h5", monitor='val_loss', verbose=1, 
            save_best_only=True, mode='auto', period=1)
    def scheduler(epoch, lr):
        # print("epoch:{}, lr:{}".format(epoch, 1e-7 * (2**epoch)))
        # return 1e-7 * (2**epoch)  # 寻找最合适的learning rate  最佳 learning rate 为 10^(-8)*(2^9)=5.12x10^(-6)
        return lr
    lr_scheduler = LearningRateScheduler(scheduler)  # 适时调整learning rate

    history = model.fit_generator(generator=trainset, validation_data=testset, \
            epochs=Config.gender_epochs, verbose=2, callbacks=[checkpoint, lr_scheduler])

    stop_time = time.time()
    print("training time:{}min".format((stop_time-start_time)/60))

    # 去除前几次迭代过大的loss, 更好展示loss下降趋势
    history_loss = {}
    for name in ("loss", "val_loss", "acc", "val_acc"):
        loss = history.history[name]
        minimum = min(loss)
        for item in loss:
            if item > minimum*5000:
                item = 0
            else:
                break
        history_loss[name] = loss

    visualize_history(history_loss)
