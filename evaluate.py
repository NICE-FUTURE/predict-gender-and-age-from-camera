# -*- "coding: utf-8" -*-

import os
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np

from datasets import BatchDataset
from models import Model


torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
np.random.seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True


def run(args):

    # 加载模型
    model = Model()
    state_dict = torch.load(args.pretrain_weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # 加载数据
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([args.img_size, args.img_size]),
                                ])
    dataset = BatchDataset(args.val_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 保存结果
    fp = open(result_save_path, "w", encoding="utf-8-sig")
    fp.write("filename,age(gt),gender(gt),age(pd),gender(pd)\n")
    with torch.no_grad():
        total = len(loader)
        for i, (inputs, ages, genders, filenames) in enumerate(loader):
            print("{:<5}/{:<5}".format(i, total), end="\r")
            inputs = inputs.to(device)
            ages = ages.to(device)
            genders = genders.to(device)

            age_pd, gender_pd = model(inputs)

            for j in range(inputs.shape[0]):
                fp.write("{},{},{},{},{}\n".format(
                    filenames[j], round(ages[j].item()*100), names[genders[j].item()], 
                    round(age_pd[j].item()*100), names[torch.argmax(gender_pd[j]).item()]
                ))
        print("{:<5}/{:<5}".format(i, total))


def metrics():
    count = 0
    total = 0
    loss = 0
    with open(result_save_path, "r", encoding="utf-8-sig") as f:
        f.readline()
        for line in f:
            total += 1
            filename, age_gt, gender_gt, age_pd, gender_pd = line.strip().split(",")
            if gender_gt == gender_pd:
                count += 1
            loss += abs(int(age_gt)-int(age_pd))
    print("loss:{}, acc:{:.6f}".format(loss, count/total))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Swin-Transformer FG simple predict:")
    parser.add_argument("--batch_size", type=int, default=8, help="training epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers parameter of dataloader")
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--pretrain_weight_path", type=str, default="", help="pretrain weight path")
    parser.add_argument("--experiment_name", type=str, required=True, help="experiment name")
    parser.add_argument("--val_dir", type=str, default="", help="val .txt file path")
    parser.add_argument("--mode", type=str, required=True, choices=["run", "metrics"])
    args = parser.parse_args()

    result_save_path = "./middle/result/{}.csv".format(args.experiment_name)
    
    with open("./data/classes.txt", "r", encoding="utf-8") as f:
        names = f.read().strip().split("\n")

    os.makedirs("./middle/result/", exist_ok=True)
    if args.mode == "run":
        assert args.pretrain_weight_path != "", "Invalid pretrain_weight_path:'{}'".format(args.pretrain_weight_path)
        assert args.val_dir != "", "Invalid val_dir:'{}'".format(args.val_dir)
        run(args)
    elif args.mode == "metrics":
        metrics()
    else:
        print("Invalid mode:{}".format(args.mode))