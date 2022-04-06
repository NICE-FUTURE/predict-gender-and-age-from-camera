# -*- "coding: utf-8" -*-

import torch
from torch.optim.lr_scheduler import LambdaLR
import math
import matplotlib.pyplot as plt


def plot_history(loss_list, acc_list, val_loss_list, val_acc_list, save_path):
    """
    绘制loss和acc曲线图
    """
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


def accuracy(outputs, targets, k=1):
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
    def __init__(self):
        self.data = dict()

    def add(self, dict_):
        for k, v in dict_.items():
            if k not in self.data:
                self.data[k] = [0.0, 0]
            self.data[k][0] += v
            self.data[k][1] += 1

    def get(self, *keys):
        avg_list = []
        for k in keys:
            if not k in self.data:
                self.data[k] = [0.0, 0]
            avg_list.append(self.data[k][0] / max(1e-5, self.data[k][1]))
        if len(avg_list) == 1:
            return avg_list[0]
        else:
            return tuple(avg_list)

    def pop(self, key):
        v = self.get(key)
        self.data[key] = [0.0, 0]
        return v

    def reset(self):
        for k in self.data.keys():
            self.data[k] = [0.0, 0]



class ListMeter:
    def __init__(self):
        self.data = dict()

    def add(self, dict_):
        for k, v in dict_.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def get(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return []

    def pop(self, key):
        v = self.get(key)
        self.data[key] = []
        return v

    def reset(self):
        for k in self.data.keys():
            self.data[k] = []