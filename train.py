# -*- "coding: utf-8" -*-

import os
import time
import logging
import traceback
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np

from datasets import BatchDataset
from models import Model
import utils




def main(args):
    # 初始化模型
    model = Model()
    model.to(device)

    with open("./model_summary.txt", "w", encoding="utf-8") as f_summary:
        print(model, file=f_summary)

    # 加载权重
    if args.pretrain_weight_path != "":
        state_dict = torch.load(args.pretrain_weight_path, map_location=device)
        model.load_state_dict(state_dict)

    # 损失函数 优化器 学习率调整器
    age_criterion = nn.MSELoss().to(device)
    gender_criterion = nn.CrossEntropyLoss().to(device)
    criterion = [age_criterion, gender_criterion]
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=[0.9, 0.999])
    scheduler = utils.WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=int(1.1*args.epochs))

    transform1 = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([args.img_size, args.img_size], antialias=True),
                                transforms.RandomPerspective(distortion_scale=0.6, p=1.0), 
                                transforms.RandomRotation(degrees=(0, 180)),
                                ])
    transform2 = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([args.img_size, args.img_size], antialias=True),
                                ])

    train_dataset = BatchDataset(args.root, args.txt_dir, "train", transform=transform1)
    val_dataset = BatchDataset(args.root, args.txt_dir, "val", transform=transform2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.info('START TIME:{}'.format(time.asctime(time.localtime(time.time()))))
    logging.info(vars(args))
    best_val = None
    meter = utils.ListMeter()
    for epoch in range(args.epochs):
        # 训练
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
        scheduler.step()

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
        meter.add({"age_loss":loss1.item(), "gender_loss":loss2.item(), "loss":loss.item(), "acc":acc})

        if i % args.log_step == 0:
            logging.info(
                "Trainning epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, total) + 
                "lr:{:.6f} ".format(optimizer.param_groups[0]['lr']) + 
                "age_loss:{:.6f} gender_loss:{:.6f} ".format(meter.get("age_loss"), meter.get("gender_loss")) + 
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

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--txt_dir", type=str, default="./data/wiki/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--pretrain_weight_path", type=str, default="", help="pretrain weight path")
    parser.add_argument("--experiment_name", type=str, required=True, help="experiment name")
    args = parser.parse_args()

    model_save_path = "./middle/models/{}-{}-best.pth".format(args.experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_path = "./middle/logs/{}-{}.log".format(args.experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    history_save_path = "./middle/history/{}-{}.png".format(args.experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        utils.fix_seed()
        main(args)
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        exit(1)