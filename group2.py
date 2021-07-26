# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:41:00 2021

@author: yigit
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import GlorotUniform, RandomUniform
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import os
import matplotlib.pyplot as plt

# This function reads images from database folder and store them into array.
def read_all_images():

    folder='Database'
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img=np.array(img)
            img=img.astype('float32')
            images.append(img)

    return images

# This function reads txt file from database folder and crop the images to get iris part.

def get_iris_crop():

    images=read_all_images()
    resultimages=[]
    img_labels=[]
    def add_iris_to_array(u,j):
        X=int(u[0])
        Y=int(u[1])
        R=int(u[2])

        result = images[j][X-R:X+R,Y-R:Y+R]
        result = cv2.resize(result,(128,128))
        resultimages.append(result)

    for i in range(len(images)):

        file = open('Database/Parameters.txt', "r")
        lines = file.readlines()[2:]


        if(i%8==0):

            for z in range(8):

                if(z<4):

                    if(z==0 or z==1):
                        u=lines[i+z].rstrip('\n').split(',')[1:]
                        name = lines[i+z].rstrip('\n').split(',')[0]
                        add_iris_to_array(u,i+z)
                    else: 
                        u=lines[i+z+2].rstrip('\n').split(',')[1:]
                        name = lines[i+z+2].rstrip('\n').split(',')[0]                        
                        add_iris_to_array(u,i+z)

                else:

                    if((z-2)==2 or (z-2)==3):
                        u=lines[i+(z-2)].rstrip('\n').split(',')[1:]
                        name = lines[i+(z-2)].rstrip('\n').split(',')[0]
                        add_iris_to_array(u,i+z)
                    else  : 
                        u=lines[i+z].rstrip('\n').split(',')[1:]
                        name = lines[i+z].rstrip('\n').split(',')[0]
                        add_iris_to_array(u,i+z)
                img_labels.append(name)
    img_labels=np.array(img_labels)
    resultimages = np.array(resultimages)

    return img_labels,resultimages

# This function is for labelling images correctly based on their names.
def determine_labels(img_names):
    count=1
    labels=[]
    for i in range(0,len(img_names),8):
        label=img_names[i:i+8]
        
        for j in range(8):
            if(j==0 or j==1 or j==4 or j==5):
                labels.append(count)
            else:
                labels.append(count+1)
        count+=2
    labels=np.array(labels)
    return labels

# The CNN Model

def CNN(X,y):
    X=X.reshape(-1, 128, 128, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50,stratify=y, shuffle=True, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.50, shuffle=True, random_state=1)
    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(128,128,1)))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(400, activation='softmax'))
    print(model.summary())
	# compile model
    opt = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
    
    batches = 32
    epochs=200
    # fit model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batches, verbose=1,callbacks=early_stop )


    _, validation_accuracy = model.evaluate(X_val, y_val, batch_size=batches)
    _, test_accuracy = model.evaluate(X_test, y_test, batch_size=batches)

    

img_names,img_crops= get_iris_crop() # Read all data and get the iris part, image_names-> names of the images, img_crops-> preprocessed images
img_labels=determine_labels(img_names) # Label the images based on their names

# One Hot Encoding fot the labels, to be able to give them to CNN model with a proper shape.
from sklearn.preprocessing import OneHotEncoder
img_labels=img_labels.reshape(-1,1)
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(img_labels).toarray()


CNN(img_crops,onehot_encoded)

