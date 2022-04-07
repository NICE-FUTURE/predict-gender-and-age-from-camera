import argparse
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models import Model


torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
np.random.seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True


def _work(image:np.ndarray) -> Tuple[int, str]:
    """predict results for one image

    Args:
        image (np.ndarray): input image (only faces)

    Returns:
        Tuple[int, str]: age, gender
    """
    inputs = transform(image).unsqueeze(0)
    inputs = inputs.to(device)

    age, gender = model(inputs)

    age = round(age[0].item()*100)
    gender = torch.argmax(gender[0]).item()
    gender = names[gender]

    return age, gender


def process_video():
    # cap = cv2.VideoCapture(0)  # read from camera
    cap = cv2.VideoCapture("./samples/sample.mp4")
    face_cascade = cv2.CascadeClassifier('H:/venvs/pytorch-cpu/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    ret, frame = cap.read()  # ret: True/False  frame: ndarray/None
    while(ret):
        ret, frame = cap.read()
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            print("Empty frame. The end.")
            break
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (x,y,w,h) in faces:

            image = frame[y:y+h, x:x+w]
            
            age, gender = _work(image)

            frame = cv2.putText(frame, "{},{}".format(gender,age), (x-10,y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA, False)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image():
    sample = Image.open("./samples/sample.png")
    face_cascade = cv2.CascadeClassifier('H:/venvs/pytorch-cpu/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    gray = np.array(sample.convert("L"))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    frame = np.array(sample)

    for (x,y,w,h) in faces:  # opencv 的 x 指宽度方向坐标

        image = frame[y:y+h, x:x+w]
        
        age, gender = _work(image)

        frame = cv2.putText(frame, "{},{}".format(gender,age), (x-10,y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA, False)
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    Image.fromarray(frame).save("./samples/sample_result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin-Transformer FG simple predict:")
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--mode", type=str, required=True, choices=["video", "image"], help="video/image")
    parser.add_argument("--pretrain_weight_path", type=str, required=True, help="pretrain weight path")
    args = parser.parse_args()

    model = Model()
    model.to(device)
    state_dict = torch.load(args.pretrain_weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with open("./data/classes.txt", "r", encoding="utf-8") as f:
        names = f.read().strip().split("\n")

    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([args.img_size, args.img_size]),
                                ])

    with torch.no_grad():
        if args.mode == "video":
            process_video()
        elif args.mode == "image":
            process_image()
        else:
            print("Invalid mode:{}".format(args.mode))
