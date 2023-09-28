import cv2
import numpy as np
import face_recognition
imgElon = face_recognition.load_image_file('images/musk1.jpeg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElon2 = face_recognition.load_image_file('images/vijay.jpg')
imgElon2 = cv2.cvtColor(imgElon2,cv2.COLOR_BGR2RGB)

face_loc2=face_recognition.face_locations(imgElon2)[0]
encodeElon2=face_recognition.face_encodings(imgElon2)[0]
face_loc=face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)
cv2.rectangle(imgElon2,(face_loc2[3],face_loc2[0]),(face_loc2[1],face_loc2[2]),(255,0,255),2)
results= face_recognition.compare_faces([encodeElon],encodeElon2)
facedis=face_recognition.face_distance([encodeElon],encodeElon2)
print(results,facedis)
cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Vijay',imgElon2)
cv2.waitKey(0)