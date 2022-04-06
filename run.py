import argparse
import time

import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models import Model


torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
np.random.seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True  # pytorch 为网络寻找最优的卷积计算方法 在网络结构不变的情况下可以提速


def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('H:/venvs/pytorch-cpu/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([args.img_size, args.img_size]),
                                ])
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            image = frame[x:x+w, y:y+h]
            image = cv2.resize(image, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
            
            inputs = transform(image)
            inputs = inputs.to(device)

            age, gender = model(image)

            age = round(age)
            gender = torch.argmax(gender)
            gender = names[gender]

            frame = cv2.putText(frame, "{},{}".format(gender,age), (x-10,y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA, False)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin-Transformer FG simple predict:")
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--pretrain_weight_path", type=str, default="", help="pretrain weight path")
    args = parser.parse_args()

    model = Model()
    model.to(device)
    state_dict = torch.load(args.pretrain_weight_path, map_location=device)
    model.load_state_dict(state_dict)

    with open("./data/classes.txt", "r", encoding="utf-8") as f:
        names = f.read().strip().split("\n")

    main()
