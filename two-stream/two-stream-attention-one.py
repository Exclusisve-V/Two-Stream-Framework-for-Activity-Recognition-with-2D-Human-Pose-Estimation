# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:09:23 2019

@author: cv
"""
import pandas as pd
import numpy as np
import cv2
import os
import glob
from sklearn.utils import shuffle

from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Dense, multiply
from keras.layers import concatenate
from keras.layers import BatchNormalization

from keras.applications.densenet import DenseNet121, DenseNet169
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet

from keras.utils.vis_utils import plot_model

jpeg_data = 'E:\Learning\person_predict\datasets\jpegs_hmdb51'
pose_data = 'E:\Learning\person_predict\datasets\pose_hmdb51'
train_path = 'E:\\Learning\\person_predict\\datasets\\hmdb51_train.csv'
test_path = 'E:\\Learning\\person_predict\\datasets\\hmdb51_test.csv'

image_size = 64
validation_size = 0.2

batch_size = 64
nb_epoch= 4

df_train = pd.read_csv(train_path).to_dict()
train_folder = df_train["sequence_name"].values()

df_test = pd.read_csv(test_path).to_dict()
test_folder = df_test["sequence_name"].values()

#print("train_folder:\n",train_folder)

train_classes=df_train['class_name']
train_sequence_name=df_train['sequence_name']

train_newDict={}
for i,j in zip(train_classes.items(),train_sequence_name.items()):
    train_newDict[j[1]]=i[1]
    
test_classes=df_test['class_name']
test_sequence_name=df_test['sequence_name']
    
test_newDict={}
for i,j in zip(test_classes.items(),test_sequence_name.items()):
    test_newDict[j[1]]=i[1]

#print("train_newDict:\n",train_newDict)    

train_strClassSet=list(set(train_newDict.values()))
cls=len(train_strClassSet)

train_clsToNum={}
for i in range(0,cls):
    train_clsToNum[train_strClassSet[i]]=i

#print("train_clsToNum:\n",train_clsToNum)i
   
def get_data(folders, newDict, date_path):
    images = []
    labels = []
    print('Going to read images')
    for folder in folders:
        #print(folder)
        strClass=newDict[folder]
        path = os.path.join(date_path, folder, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            label=np.zeros(cls)
            label[train_clsToNum[strClass]]=1.0
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


train_images, train_images_labels = get_data(train_folder, train_newDict, jpeg_data)
print("train_images:", train_images.shape)

test_images, test_images_labels = get_data(test_folder, test_newDict, jpeg_data)
print("test_images:", test_images.shape)

train_pose, train_pose_labels = get_data(train_folder, train_newDict, pose_data)
print("train_pose:", train_images.shape)

test_pose, test_pose_labels = get_data(test_folder, test_newDict, pose_data)
print("test_pose:", test_images.shape)

x1_train, x2_train, y_train = shuffle(train_images, train_pose, train_images_labels)
x1_test, x2_test, y_test = shuffle(test_images, test_pose, test_images_labels)


def multi_input_model():

    input1 = Input(shape=(image_size,image_size,3), name='Input1')
    input2 = Input(shape=(image_size,image_size,3), name='Input2') 
    
    x1 = MobileNet(include_top=False, weights=None,
               input_tensor=None, 
               input_shape=(image_size,image_size,3),
               pooling='avg')(input1)
 
    x2 = MobileNetV2(include_top=False, weights=None,
               input_tensor=None, 
               input_shape=(image_size, image_size, 3),
               pooling='avg')(input2)
    
    x = concatenate([x1, x2], name='Concatenate')
    
    weight = Dense(512, activation='softmax', name='Weight')(x) 
    
    x = Dense(512,activation='relu')(x) 
    
    x = multiply([x, weight], name='Multiply')

    x = BatchNormalization(name='BN')(x)
    
    output = Dense(cls, activation='softmax', name='Output')(x)
 
    model = Model(inputs=[input1, input2], outputs=[output])
    model.summary()
    
    return model

model = multi_input_model()

plot_model(model, 'Multi_input_model_attention.png', show_shapes=True)
 

early_stopping = EarlyStopping(monitor='val_acc', min_delta= 0,
                               patience=10, verbose=1, mode='auto',
                               baseline=None, restore_best_weights=False)

filepath = "attention-weights-best"+str(-nb_epoch)+".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True,
                             mode='auto')

csv_logger = CSVLogger('attention-train'+str(-nb_epoch)+'.csv')

callbacks_list = [early_stopping, checkpoint, csv_logger]

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit([x1_train, x2_train], y_train, 
                    batch_size=batch_size, 
                    epochs=nb_epoch,
                    verbose=1,
                    callbacks=callbacks_list, 
                    validation_split=0.2,
                    shuffle=True)
        
model.save('my-attention-model'+str(-nb_epoch)+'.h5')

score = model.evaluate([x1_test, x2_test], y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_ture=np.argmax(y_test,axis=1)
#print(classes_true)
#y_test=y_test.max(axis=1)
classes_true=pd.Series(y_ture)
classes_true.to_csv('classes_true.csv',index = False)

#print(model.predict(x_test))		# 打印概率
classes = np.argmax(model.predict([x1_test, x2_test]), axis=1)
classes_predict=pd.Series(classes)
classes_predict.to_csv('classes_predict.csv',index = False)

