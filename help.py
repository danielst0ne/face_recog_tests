import face_recognition
import cv2
import numpy as np
import glob
import os
known_face_encodings = []
known_face_names = []
imglistfull = glob.glob("img/*.jpg")
imglist = [os.path.basename(img)[:-4] for img in imglistfull]
for img,name in zip(imglistfull,imglist):
  face_image= face_recognition.load_image_file(str(img))
  face_encoding = face_recognition.face_encodings(face_image)[0]
  known_face_encodings.append(face_encoding)
  known_face_names.append(name)
print (known_face_names,known_face_encodings)