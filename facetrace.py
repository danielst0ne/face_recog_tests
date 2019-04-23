import cv2
import face_recognition
import numpy
from PIL import Image, ImageDraw
import json

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()

else:

    rval = False

while rval:
  # cv2.imshow("preview", frame)
	rval, frame = vc.read()
  #do all processing under here
	pil_image = Image.fromarray(frame).convert('RGB')
  
	d = ImageDraw.Draw(pil_image)
	
	face_landmarks_list = face_recognition.face_landmarks(frame)

	outfile = open('daniel.txt','w')
	
	json.dump(face_landmarks_list,outfile)

  #print(face_landmarks_list)
	
	for landmarks in face_landmarks_list:
		for feature in landmarks.keys():
			d.line(landmarks[feature],width = 2)




	open_cv_image = numpy.array(pil_image)
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	cv2.imshow("preview", open_cv_image)
 


	key = cv2.waitKey(20)
	if key == 27: # exit on ESC
		break

'''
[{'chin': [(482, 285), (488, 324), (498, 363), (509, 400), (525, 434), (551, 462), (585, 481), (619, 497), (655, 500), (688, 491), (716, 469), (741, 444), (758, 413), (767, 376), (770, 339), (774, 300), (774, 261)], 
'left_eyebrow': [(506, 263), (525, 247), (552, 242), (580, 243), (605, 251)], 
'right_eyebrow': [(648, 246), (674, 235), (702, 230), (730, 231), (751, 244)], 
'nose_bridge': [(631, 274), (633, 298), (635, 322), (638, 348)], 
'nose_tip': [(611, 365), (625, 370), (639, 373), (653, 368), (666, 362)], 
'left_eye': [(544, 281), (560, 272), (579, 271), (597, 283), (579, 288), (560, 289)], 
'right_eye': [(667, 277), (681, 263), (700, 261), (718, 268), (704, 278), (685, 280)], 
'top_lip': [(588, 410), (608, 403), (626, 399), (641, 402), (656, 397), (676, 399), (698, 402), (688, 404), (657, 409), (642, 413), (627, 411), (597, 411)], 
'bottom_lip': [(698, 402), (679, 421), (661, 429), (645, 432), (629, 431), (610, 426), (588, 410), (597, 411), (628, 412), (643, 413), (658, 410), (688, 404)]}]


'''
cv2.destroyWindow("preview")