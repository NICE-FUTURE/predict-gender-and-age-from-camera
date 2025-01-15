import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from models import Model


def process_single_frame(face_cascade, frame:np.ndarray, padding_ratio=0.1, offset_ratio=0.1) -> np.ndarray:
    """ 
    Args:
        face_cascade: face detector
        frame (np.ndarray): BGR image
        padding_ratio (float, optional): Padding margin in result image. Defaults to 0.1
        offset_ratio (float, optional): Offset ratio of face boxes. Defaults to 0.1

    Returns:
        result image (np.ndarray): BGR image
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ### Expand the boundary of the original image to prevent the text at the boundary position from disappearing
    render_frame = frame.copy()
    x_padding, y_padding = int(width * padding_ratio), int(height * padding_ratio)
    render_frame = cv2.copyMakeBorder(render_frame, y_padding, y_padding, x_padding, x_padding, cv2.BORDER_CONSTANT, value=(0,0,0))

    for (x,y,w,h) in faces:  # x: width direction
        x_offset, y_offset = int(width * offset_ratio), int(height * offset_ratio)
        x, y = x - x_offset, y - y_offset
        w, h = w + 2 * x_offset, h + 2 * y_offset
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        face_img = np.ascontiguousarray(frame[y:y+h, x:x+w][:,:,::-1])  # BGR to RGB
        inputs = transform(face_img).unsqueeze(0).to(device)
        age, gender = model(inputs)
        age = round(age[0].item()*100)
        gender = torch.argmax(gender[0]).item()
        gender = names[gender]
        x, y = x + x_padding, y + y_padding  # don't forget move the box point
        render_frame = cv2.putText(render_frame, "{},{}".format(gender,age), (x-10,y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA, False)
        render_frame = cv2.rectangle(render_frame,(x,y),(x+w,y+h),(255,0,0),2)

    return render_frame


def process_video(camera=False):
    if camera:
        cap = cv2.VideoCapture(0)  # read from camera
    else:
        cap = cv2.VideoCapture("./samples/sample.mp4")
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    while(True):
        ret, frame = cap.read()  # ret: True/False  frame: ndarray/None
        if ret:
            frame = process_single_frame(face_cascade, frame)  # provide RGB instead of BGR
            cv2.imshow('frame', frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # wait 1 millisecond and check if 'Q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image():
    image = np.array(Image.open("./samples/sample.png").convert("RGB"))[:,:,::-1]  # RGB to BGR
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    result = process_single_frame(face_cascade, image)[:,:,::-1]  # BGR to RGB
    Image.fromarray(result).save("./samples/sample_result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["video", "image", "camera"], help="video/image/camera mode")
    parser.add_argument("--weights", type=str, required=True, help="pretrain weight path")

    parser.add_argument("--img_size", type=int, default=224, help="image size")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(timm_pretrained=False)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with open("./data/wiki/classes.txt", "r", encoding="utf-8") as f:
        names = f.read().strip().split("\n")

    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize([args.img_size, args.img_size], antialias=True),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])

    with torch.no_grad():
        if args.mode == "video":
            process_video()
        elif args.mode == "camera":
            process_video(camera=True)
        elif args.mode == "image":
            process_image()
        else:
            print("Invalid mode:{}".format(args.mode))
