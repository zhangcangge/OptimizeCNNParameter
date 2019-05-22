import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "./elephant-vs-horse"
GATEGORIES = ['elephant', 'horse']

training_data = []
IMG_SIZE = 50

def create_training_data():
    for category in GATEGORIES:
        print(category)
        path = os.path.join(DATADIR, category)
        class_num = GATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

import random

random.shuffle(training_data)

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open("XElephantVSHorse.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("YElephantVSHorse.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()