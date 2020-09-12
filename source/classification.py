##--- Basic imports ---

import cv2
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import load_model
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


##--- Global variables ---

convert_array = {'Happy' : 0,'Sad' : 1,'Fear' : 2 }
convert_reverse = {0 : 'Happy', 1 : 'Sad', 2 : 'Fear'}


##-- preprocessing ---

def preprocess_train(trainfile):
    
    data = pd.read_csv(trainfile)

    X = []
    y_cat= []
    
    for tup in data.itertuples():
        image = np.array(tup[-2304:]).reshape(48,48).astype(float).tolist()
        X.append(image)
        y_cat.append(tup[1])
    
    X = np.array(X)


    ##--- Convert Gray scale to RGB (by repeating the same channel) ---

    X_rgb = []
    for item in X:
          X_rgb.append(np.stack((item,)*3,axis = -1).tolist())
    X_rgb = np.array(X_rgb)

    dim = (80, 80)

    temp  = X_rgb
    X = []
    for i in range(len(temp)):
        X.append(cv2.resize(temp[i],dim))
    X = np.array(X) 

    ##--- oversampling ---

    train_dic = {}
    for i in range(len(y_cat)):
        try:
            train_dic[y_cat[i]].append(X[i])
        except:  
            train_dic[y_cat[i]] = [X[i]]


    m = max([len(train_dic[key]) for key in train_dic.keys()])
    for key in train_dic.keys():
        l = (m - len(train_dic[key]))
        if l!= 0:
            X = np.concatenate((X, np.array(random.choices(train_dic[key],k=l))))
            y_cat = np.concatenate((y_cat,np.array([key]*l)),axis=0)
            

    y_encoded = np.array([convert_array[i] for i in y_cat])
    y = np.eye(len(set(y_encoded)))[y_encoded]

    return X, y


def preprocess_test(testfile):
    
    data = pd.read_csv(testfile)
    
    X = []

    for tup in data.itertuples():
        image = np.array(tup[-2304:]).reshape(48,48).astype(float).tolist()
        X.append(image)

    X = np.array(X)


    ##--- Convert Gray scale to RGB (by repeating the same channel) ---

    X_rgb = []
    for item in X:
          X_rgb.append(np.stack((item,)*3,axis = -1).tolist())
    X_rgb = np.array(X_rgb)

    dim = (80, 80)

    temp  = X_rgb
    X = []
    for i in range(len(temp)):
        X.append(cv2.resize(temp[i],dim))
    X = np.array(X) 

    return X


##--- core functions ---

def train_a_model(trainfile):
    
    X, y = preprocess_train(trainfile)
    
    ##--- Define model ---
    
    base_model = VGG16(include_top=False, input_shape=(80, 80, 3), weights="imagenet") 
 
    base_model.trainable = True
    
    model = Sequential([base_model])
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.25))
    model.add(Dense(3, activation = "softmax"))

    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    rlr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience= 3, verbose=1, mode='auto', min_lr=0)

    
    history = model.fit(X, y,
              batch_size=64,
              epochs=50,
              shuffle=True, verbose = 2, callbacks=[rlr])

    return model
  
def test_the_model(model, testfile):

    X = preprocess_test(testfile)
    
    # making prediction
    prediction = model.predict(X)
    
    final_prediction = [convert_reverse[np.argmax(x)] for x in prediction]
    
    return final_prediction
