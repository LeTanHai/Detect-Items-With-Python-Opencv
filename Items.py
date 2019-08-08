import cv2
import numpy as np
import pickle

watch_cascade = cv2.CascadeClassifier('cascades/data/cascade_watch.xml')
casio_cascade = cv2.CascadeClassifier('cascades/data/Casio_cascade.xml')
iphone_cascade = cv2.CascadeClassifier('cascades/data/iphone_cascade.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Train/img_train.yml')

lables = {'name':1}

with open('Lables-id/Items-lable.pickle','rb') as f:
	total_lables = pickle.load(f)
	lables = {v:k for k,v in total_lables.items()}

cap = cv2.VideoCapture(0)

subtrac = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(3,3),5)
    img_sub = subtrac.apply(blur)
    # để tìm được biên đầu tiên phải sử dụng thresold or canny để tìm ra các cạnh
    canny = cv2.Canny(blur,50,150)
    contour,h = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    watch = watch_cascade.detectMultiScale(gray, scaleFactor = 4, minNeighbors = 3)

    casio = casio_cascade.detectMultiScale(gray, scaleFactor = 10, minNeighbors = 10)

    iphone = iphone_cascade.detectMultiScale(gray, scaleFactor = 8, minNeighbors = 8)

    #for c in (contour):	+
    #    (x,y,w,h) = cv2.boundingRect(c)

    #    if w < 200 or h < 200:
    #	    continue
    	    
    #	    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    #	    img_cut_gray = gray[y:y+h, x:x+w]
    #	    cv2.imwrite('img_result',img_cut_gray)
    for (x, y, w, h) in watch:
    	roi_gray_watch = gray[y:y+h, x:x+w]
    	cv2.imwrite('image_cut.png',roi_gray_watch)
    	_id,conf = recognizer.predict(roi_gray_watch)
    	if conf > 90 and conf <= 100:
    		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = lables[_id]
    		color = (255,255,0)
    		stroke = 2
    		cv2.putText(frame,name,(x,y), font, 1, color, stroke, cv2.LINE_AA)
    		print(_id)

    cv2.imshow('result',frame)
    key = cv2.waitKey(5) & 0xFF
    if key ==27:
    	break

cap.release()
cv2.destroyAllWindows()