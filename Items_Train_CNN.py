import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle

DATADIR = "C:\\Users\\tetan\\OneDrive\\Documents\\PyThon\\Detect-Items"

CATEGORIES = ["Watch","TanHai"]

IMG_SIZE = 300

training_data = []

X = []
y = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats
        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        print(class_num)

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
            	print(img) # tên của ảnh
            	img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            	new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            	training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()
print(len(training_data))

for features,label in training_data:
    X.append(features)  # ma trận pixel của ảnh
    y.append(label)     # class_num 0 đại diện cho watch

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1)) # test thử X[0]

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(X)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()