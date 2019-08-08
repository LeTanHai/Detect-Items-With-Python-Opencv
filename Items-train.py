import cv2
import numpy as np
import os
from PIL import Image
import pickle
import time

dic_lables_id = {}
current_id = 0
img_train = []
id_train = []

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir,'Image')

watch_cascade = cv2.CascadeClassifier('cascades/data/cascade_watch.xml')
casio_cascade = cv2.CascadeClassifier('cascades/data/Casio_cascade.xml')
iphone_cascade = cv2.CascadeClassifier('cascades/data/iphone_cascade.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

for root,dirs,files in os.walk(image_dir):
	for file in files:
		if file.endswith('jpg') or file.endswith('png'):
			path = os.path.join(root,file)
			lables = os.path.basename((os.path.dirname(path))).lower()

			if not lables in dic_lables_id:
				dic_lables_id[lables] = current_id
				current_id += 1
			_id = dic_lables_id[lables]
			#id_train.append(_id)
			#
			frame = cv2.imread(path)
			gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			#gray_image = Image.open(path).convert('L')
			#image_array = np.array(gray_image,'uint8')
			
			#blur = cv2.GaussianBlur(image_array,(3,3),5)
			#canny = cv2.Canny(blur,50,150)

			#contour,h = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

			#for c in (contour):
			#	(x,y,w,h) = cv2.boundingRect(c)

			#	if w < 200 or h < 200:
			#		continue

			#	roi = image_array[y:y+h,x:x+w]
			#	img_train.append(roi)
			#	id_train.append(_id)
			#
			watch = watch_cascade.detectMultiScale(gray, scaleFactor=4, minNeighbors=3) #hàm phát hiện ra vị trí khuôn mặt trên ảnh
			for (x,y,w,h) in watch:
				roi = gray[y:y+h, x:x+w]
				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
				img_array = np.array(roi,'uint8')
				img_train.append(img_array)
				id_train.append(_id)
			cv2.imshow('result',roi)
			cv2.imwrite('image_cut.png',roi)


print(dic_lables_id)	
with open('Lables-id/Items-lable.pickle','wb') as f:
	pickle.dump(dic_lables_id,f)

recognizer.train(img_train,np.array(id_train))
recognizer.save('Train/img_train.yml')

cv2.waitKey(0)