import cv2
from keras.models import load_model
import time

gender_model = load_model("cnn4gender.h5")
age_model = load_model("cnn4age.h5")

def get_frame():
    cap = cv2.VideoCapture(0)
    # 使用cv2自带的人脸识别
    face_cascade = cv2.CascadeClassifier('/your/path/to/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            image = gray[x:x+w, y:y+h]
            try:
                image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_AREA)
            except:
                continue
            
            gender = get_gender(image)
            age = get_age(image)
            # print("{}: {},{}".format(temp, gender,age))

            frame = cv2.putText(frame, "{},{}".format(gender,age), (x-10,y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA, False)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def get_gender(image):
    image = image.reshape(1, 20, 20, 1)
    gender = gender_model.predict(image)
    if gender[0] == 1:
        return 'male'
    else:
        return 'female'

def get_age(image):
    image = image.reshape(1, 20, 20, 1)
    age = age_model.predict(image)
    age = round(age[0][0])
    return age

if __name__ == "__main__":
    get_frame()
