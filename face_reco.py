import cv2
import numpy as np
import face_recognition
import os

path = 'images'
images = []
classnames = []
mylist = os.listdir(path)

for cls in mylist:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classnames.append(os.path.splitext(cls)[0])
print(classnames)
def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown = findEncodings(images)
print(len(encodelistknown))
""" imgElon = face_recognition.load_image_file('images/musk1.jpeg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElon2 = face_recognition.load_image_file('images/vijay.jpg')
imgElon2 = cv2.cvtColor(imgElon2,cv2.COLOR_BGR2RGB) """