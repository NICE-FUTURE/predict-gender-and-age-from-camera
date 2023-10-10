# -*- "coding: utf-8" -*-

import os
import argparse
from datetime import datetime

import torch
import torch.utils.data
import torchvision.transforms as transforms

from datasets import BatchDataset
from models import Model
import utils


def eval():

    # 加载模型
    model = Model(timm_pretrained=False)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # 加载数据
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([args.img_size, args.img_size], antialias=True),
                                ])
    dataset = BatchDataset(args.root, args.txt_dir, "val", transform=transform)
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
            loss += (int(age_gt)/100.0 - int(age_pd)/100.0)**2
    print("loss:{}, acc:{:.6f}".format(loss/total, count/total))  # 此处离线评估的loss会比训练期间的验证集更小 因为保存csv时用round做了四舍五入取整


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--txt_dir", type=str, default="./data/wiki/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--weight", type=str, default="", help="pretrain weight path")
    parser.add_argument("--experiment_name", type=str, required=True, help="experiment name")
    parser.add_argument("--mode", type=str, required=True, choices=["eval", "metrics"])
    args = parser.parse_args()

    result_save_path = "./middle/result/{}.csv".format(args.experiment_name)

    os.makedirs("./middle/result/", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.fix_seed()
    if args.mode == "eval":
        assert os.path.exists(args.root), f"Dataset path '{args.root}' NOT exists."
        assert os.path.exists(args.txt_dir), f"*.txt path '{args.txt_dir}' NOT exists."
        assert os.path.exists(args.weights), f"Weights path '{args.weights}' NOT exists."
        with open(os.path.join(args.txt_dir, "classes.txt"), "r", encoding="utf-8") as f:
            names = f.read().strip().split("\n")
        eval()
    elif args.mode == "metrics":
        assert os.path.exists(result_save_path), f"CSV path '{result_save_path}' NOT exists."
        metrics()
    else:
        print("Invalid mode:{}".format(args.mode))
