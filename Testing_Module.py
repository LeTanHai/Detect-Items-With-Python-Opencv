import cv2
import tensorflow as tf
import pickle


watch_cascade = cv2.CascadeClassifier('cascades/data/cascade_watch.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Train/img_train.yml')

CATEGORIES = ["Watch", "TanHai"]

def prepare(filepath):
    IMG_SIZE = 70  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("64x3-CNN.model")

#print(prediction)  # will be a list in a list.
#print(CATEGORIES[int(prediction[0][0])])

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    watch = watch_cascade.detectMultiScale(gray, scaleFactor = 4, minNeighbors = 3)

    for (x, y, w, h) in watch:
    	roi_gray_watch = gray[y:y+h, x:x+w]
    	cv2.imwrite('image_cut.png',roi_gray_watch)
   
    prediction = model.predict([prepare('image_cut.png')])

    font = cv2.FONT_HERSHEY_SIMPLEX
    name = CATEGORIES[int(prediction[0][0])]
    color = (255,255,0)
    stroke = 2
    cv2.putText(frame,name,(x,y), font, 1, color, stroke, cv2.LINE_AA)

    cv2.imshow('result',frame)
    key = cv2.waitKey(5) & 0xFF
    if key ==27:
    	break

cap.release()
cv2.destroyAllWindows()