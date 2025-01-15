from datetime import datetime
import math
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR


def fix_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cal_eta(start_time, cur, total):
    time_now = datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total-cur) / float(cur)
    delta = (time_now - start_time)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(eta)


def plot_history(metrics, save_path):
    """
    绘制loss和acc曲线图
    """
    matplotlib.use("agg")
    loss_list, acc_list, val_loss_list, val_acc_list = metrics
    plt.figure(figsize=(10,8))
    plt.subplot(211)
    plt.plot(loss_list, color="blue", label="loss")
    plt.plot(val_loss_list, color="orange", label="val_loss")
    plt.legend()
    plt.subplot(212)
    plt.plot(acc_list, color="blue", label="acc")
    plt.plot(val_acc_list, color="orange", label="val_acc")
    plt.legend()

    plt.savefig(save_path)
    plt.close()


def cal_accuracy(outputs, targets, k=1):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    if len(outputs.shape) == 1:
        outputs = torch.unsqueeze(outputs, axis=0)
    batch_size = targets.size(0)
    _, ind = outputs.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor

    return correct_total.item() / batch_size


# 代码片段取自 https://github.com/TACJu/TransFG/blob/master/utils/scheduler.py
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]   # type: ignore
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


class ListMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = []

    def add(self, dict_):
        for k, v in dict_.items():
            if k not in self.__data:
                self.__data[k] = []
            self.__data[k].append(v)

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]]
        else:
            v_list = [self.__data[k] for k in keys]
            return tuple(v_list)

    def get_mean(self, *keys):
        if len(keys) == 1:
            return np.mean(self.__data[keys[0]])
        else:
            v_list = [np.mean(self.__data[k]) for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = []
        else:
            v = self.get(key)
            self.__data[key] = []
            return v
