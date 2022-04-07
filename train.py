# -*- "coding: utf-8" -*-

from datetime import datetime
import os
import time
import logging
import traceback
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np

from datasets import BatchDataset
from models import Model
import utils


torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
np.random.seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True


def main(args):

    # 初始化模型
    model = Model()
    model.to(device)
    
    # 加载权重
    if args.pretrain_weight_path != "":
        state_dict = torch.load(args.pretrain_weight_path, map_location=device)
        model.load_state_dict(state_dict)

    with open("./model_summary.txt", "w", encoding="utf-8") as f_summary:
        print(model, file=f_summary)

    # 损失函数 优化器 学习率调整器
    age_criterion = nn.MSELoss().to(device)
    gender_criterion = nn.CrossEntropyLoss().to(device)
    criterion = [age_criterion, gender_criterion]
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = utils.WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=int(1.1*args.epochs))

    transform1 = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([args.img_size, args.img_size]),
                                transforms.RandomPerspective(distortion_scale=0.6, p=1.0), 
                                transforms.RandomRotation(degrees=(0, 180)),
                                ])
    transform2 = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([args.img_size, args.img_size]),
                                ])

    train_dataset = BatchDataset(args.train_dir, transform=transform1)
    val_dataset = BatchDataset(args.val_dir, transform=transform2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    logging.info('START TIME:{}'.format(time.asctime(time.localtime(time.time()))))
    logging.info(vars(args))
    best_val = None
    meter = utils.ListMeter()
    for epoch in range(args.epochs):
        # 训练
        scheduler.step()
        loss, acc = train(train_loader, model, criterion, optimizer, epoch, args)
        if np.isnan(loss):
            print("ERROR! Loss is Nan. Break.")
            break
        meter.add({"loss": loss, "acc": acc})
        # 验证
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, args)
        meter.add({"val_loss": val_loss, "val_acc": val_acc})
        logging.info(
            "[Epoch:{:<5}/{:<5}] ".format(epoch+1, args.epochs) + 
            "lr:{:.6f} ".format(optimizer.param_groups[0]['lr']) + 
            "loss:{:.6f} val_loss:{:.6f} ".format(loss, val_loss) + 
            "acc:{:.6f} val_acc:{:.6f}".format(acc, val_acc)
        )
        utils.plot_history(meter.get("loss"), meter.get("acc"), meter.get("val_loss"), meter.get("val_acc"), history_save_path)

        # 保存
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_save_path)
            logging.info("Saved best model.")

    utils.plot_history(meter.pop("loss"), meter.pop("acc"), meter.pop("val_loss"), meter.pop("val_acc"), history_save_path)
    logging.info('STOP TIME:{}'.format(time.asctime(time.localtime(time.time()))))


def train(train_loader, model, criterion, optimizer, epoch, args):
    age_criterion, gender_criterion = criterion
    model.train()
    meter = utils.AverageMeter()
    total = len(train_loader)
    for i, (inputs, ages, genders, filenames) in enumerate(train_loader):
        inputs = inputs.to(device)
        ages = ages.to(device)
        genders = genders.to(device)

        age_pd, gender_pd = model(inputs)

        loss1 = age_criterion(age_pd, ages)
        loss2 = gender_criterion(gender_pd, genders)
        loss = loss1 + loss2
        acc = utils.accuracy(gender_pd, genders)
        meter.add({"age_loss":loss1, "gender_loss":loss2, "loss":loss.item(), "acc":acc})

        if i % args.log_step == 0:
            logging.info(
                "Trainning epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, total) + 
                "lr:{:.6f} ".format(optimizer.param_groups[0]['lr']) + 
                "age_loss:{:.6f} gender_loss:{:.6f}".format(meter.get("age_loss"), meter.get("gender_loss")) + 
                "loss:{:.6f} acc:{:.6f}".format(meter.get("loss"), meter.get("acc"))
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return meter.pop("loss"), meter.pop("acc")


def validate(val_loader, model, criterion, epoch, args):
    age_criterion, gender_criterion = criterion
    model.eval()
    meter = utils.AverageMeter()
    with torch.no_grad():
        total = len(val_loader)
        for i, (inputs, ages, genders, filenames) in enumerate(val_loader):
            inputs = inputs.to(device)
            ages = ages.to(device)
            genders = genders.to(device)

            age_pd, gender_pd = model(inputs)

            loss1 = age_criterion(age_pd, ages)
            loss2 = gender_criterion(gender_pd, genders)
            loss = loss1 + loss2
            acc = utils.accuracy(gender_pd, genders)
            meter.add({"loss": loss.item(), "acc": acc})

            if i % args.log_step == 0:
                logging.info(
                    "Validating epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, total) + 
                    "loss:{:.6f} acc:{:.6f}".format(meter.get("loss"), meter.get("acc"))
                )

    return meter.pop("loss"), meter.pop("acc")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Swin-Transformer FG simple predict:")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="training epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers parameter of dataloader")
    parser.add_argument("--log_step", type=int, default=50, help="log accuracy each log_step batchs")
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--lr", type=float, default=0.001, help="backbone initial learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10, help="use warmup cosine schedule")
    parser.add_argument("--pretrain_weight_path", type=str, default="", help="pretrain weight path")
    parser.add_argument("--experiment_name", type=str, required=True, help="experiment name")
    parser.add_argument("--train_dir", type=str, required=True, help="train .txt file path")
    parser.add_argument("--val_dir", type=str, required=True, help="val .txt file path")
    args = parser.parse_args()

    model_save_path = "./middle/models/{}-best.pth".format(args.experiment_name)
    log_path = "./middle/logs/{}-{}.log".format(args.experiment_name, datetime.now()).replace(":",".")
    history_save_path = "./middle/history/{}-{}.png".format(args.experiment_name, datetime.now()).replace(":",".")
    
    os.makedirs("./middle/models/", exist_ok=True)
    os.makedirs("./middle/logs/", exist_ok=True)
    os.makedirs("./middle/history/", exist_ok=True)
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode='a'), logging.StreamHandler()]
    )
    try:
        main(args)
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        exit(1)