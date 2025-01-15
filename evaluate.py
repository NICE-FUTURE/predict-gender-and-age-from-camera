# -*- "coding: utf-8" -*-

import json
import os
import argparse

import torch
import torch.utils.data
import torchvision.transforms as transforms

from datasets import BatchDataset
from models import Model
import utils


def test():

    # 加载模型
    model = Model(timm_pretrained=False)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 加载数据
    transform = transforms.Compose([
                                transforms.Resize([args.img_size, args.img_size], antialias=True),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    dataset = BatchDataset(args.root, args.txt_dir, "test", transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
    )

    # 保存结果
    fp = open(result_save_path, "w", encoding="utf-8-sig")
    fp.write("filename,age(truth),gender(truth),age(pred),gender(pred)\n")
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
            loss += (int(age_gt)/100.0 - int(age_pd)/100.0)**2  # MSE # 此处离线评估的loss会受保存csv时的四舍五入取整操作影响
    result = {
        "loss": loss/total, 
        "acc": round(count/total, 6),
    }
    with open(result_score_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True, help="pretrain weight path")
    parser.add_argument("--skip_test", action="store_true", help="skip test, only calculate metrics")

    parser.add_argument("--txt_dir", type=str, default="./data/wiki/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    assert os.path.exists(args.root), f"Dataset folder '{args.root}' NOT exists."
    assert os.path.exists(args.weights), f"Weights file '{args.weights}' NOT exists."
    experiment_name = os.path.basename(args.weights).split(".")[0]
    result_save_path = "./middle/result/{}.csv".format(experiment_name)
    result_score_path = "./middle/result/{}.json".format(experiment_name)
    os.makedirs("./middle/result/", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.fix_seed()

    # 加载类别名称
    with open(os.path.join(args.txt_dir, "classes.txt"), "r", encoding="utf-8") as f:
        names = f.read().strip().split("\n")

    if not args.skip_test:
        # 生成测试集结果
        test()
    # 计算指标
    metrics()
