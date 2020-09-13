import cv2
import numpy as np
from keras.models import load_model

import time

from config import Config

gender_model = load_model("cnn4gender_best.h5")
age_model = load_model("cnn4age_best.h5")
wait = 0

def get_frame():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('C:/Users/23755/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            image = frame[x:x+w, y:y+h]
            try:
                image = cv2.resize(image, (Config.height, Config.width), interpolation=cv2.INTER_AREA)
            except:
                continue
            
            gender = get_gender(image)
            age = get_age(image)
            # age = age*(Config.max_age-Config.min_age) + Config.min_age
            # print("{},{}".format(gender,age))

            frame = cv2.putText(frame, "{},{}".format(gender,age), (x-10,y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA, False)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(faces) > 0:
            time.sleep(wait)

    cap.release()
    cv2.destroyAllWindows()

def get_gender(image):
    image = np.expand_dims(image, axis=0)
    gender = gender_model.predict(image)
    if gender[0] == 1:
        return 'male'
    else:
        return 'female'

def get_age(image):
    image = np.expand_dims(image, axis=0)
    age = age_model.predict(image)
    age = round(age[0][0])
    return age

if __name__ == "__main__":
    get_frame()
