import argparse
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models import Model


def process_single_frame(face_cascade, frame:np.ndarray, padding_ratio=0.1) -> np.ndarray:
    height, width = frame.shape[:2]
    gray = np.array(Image.fromarray(frame, mode="RGB").convert("L"))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ### Expand the boundary of the original image to prevent the text at the boundary position from disappearing
    render_frame = frame.copy()
    x_padding, y_padding = int(width * padding_ratio), int(height * padding_ratio)
    render_frame = cv2.copyMakeBorder(render_frame, y_padding, y_padding, x_padding, x_padding, cv2.BORDER_CONSTANT, value=(0,0,0))

    for (x,y,w,h) in faces:  # x: width direction
        ### extend box margin; the offset ratio in wiki_crop is 40%
        # x_offset, y_offset = int(width * 0.4), int(height * 0.4)
        x_offset, y_offset = int(width * 0.1), int(height * 0.1)
        x, y = x - x_offset, y - y_offset
        w, h = w + 2 * x_offset, h + 2 * y_offset
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y

        inputs = transform(frame[y:y+h, x:x+w]).unsqueeze(0)
        inputs = inputs.to(device)
        age, gender = model(inputs)
        age = round(age[0].item()*100)
        gender = torch.argmax(gender[0]).item()
        gender = names[gender]
        x, y = x + x_padding, y + y_padding  # don't forget move the box point
        render_frame = cv2.putText(render_frame, "{},{}".format(gender,age), (x-10,y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA, False)
        render_frame = cv2.rectangle(render_frame,(x,y),(x+w,y+h),(0,0,255),2)

    return render_frame


def process_video():
    # cap = cv2.VideoCapture(0)  # read from camera
    cap = cv2.VideoCapture("./samples/sample.mp4")
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    while(True):
        ret, frame = cap.read()  # ret: True/False  frame: ndarray/None
        if ret:
            frame = process_single_frame(face_cascade, np.ascontiguousarray(frame[:,:,::-1]))  # provide RGB instead of BGR
            cv2.imshow('frame', frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # wait 1 millisecond and check if 'Q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image():
    image = np.array(Image.open("./samples/sample.png"))
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    image = process_single_frame(face_cascade, image)
    Image.fromarray(image).save("./samples/sample_result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin-Transformer FG simple predict:")
    parser.add_argument("--mode", type=str, required=True, choices=["video", "image"], help="video/image")
    parser.add_argument("--weights", type=str, required=True, help="pretrain weight path")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(timm_pretrained=False)
    model.to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with open("./data/wiki/classes.txt", "r", encoding="utf-8") as f:
        names = f.read().strip().split("\n")

    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([args.img_size, args.img_size], antialias=True),
                                ])

    with torch.no_grad():
        if args.mode == "video":
            process_video()
        elif args.mode == "image":
            process_image()
        else:
            print("Invalid mode:{}".format(args.mode))
