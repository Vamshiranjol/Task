import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
import io
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import  tqdm
from sklearn.utils import shuffle

base_dir=r"C:\Users\kanis\Downloads\archive (2)"
for i in os.listdir(base_dir):
    folderPath=os.path.join(base_dir,i)
    for j in os.listdir(folderPath):
        print(os.path.join(folderPath,j))


X_train=[]
y_train=[]
image_size=150
labels=['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
for i in labels:
    folderPath=os.path.join(r'C:\Users\kanis\Downloads\archive (2)\Training',i)
    for j in os.listdir(folderPath):
        img=cv2.imread(os.path.join(folderPath,j))
        img=cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)

for i in labels:
    folderPath=os.path.join(r'C:\Users\kanis\Downloads\archive (2)\Testing',i)
    for j in os.listdir(folderPath):
        img=cv2.imread(os.path.join(folderPath,j))
        img=cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)

X_train=np.array(X_train)
y_train=np.array(y_train)

X_train,y_train=shuffle(X_train,y_train,random_state=101)
X_train.shape

X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.1,random_state=101)

y_train_new=[]
for i in y_train:
    y_train_new.append(labels.index(i))
y_train=y_train_new
y_train=tf.keras.utils.to_categorical(y_train)  

y_test_new=[]
for i in y_train:
    y_test_new.append(labels.index(i))
y_test=y_test_new
y_test=tf.keras.utils.to_categorical(y_test)  

