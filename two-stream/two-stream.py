# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:45:10 2019

@author: cv
"""

import pandas as pd
import numpy as np
import cv2
import os
import glob
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from keras import Input, Model
from keras.layers import Dense, concatenate, BatchNormalization

from keras.applications.densenet import DenseNet121, DenseNet169
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet

from keras.utils.vis_utils import plot_model

jpeg_path = 'E:\Learning\person_predict\datasets\jpegs_hmdb51'
pose_path = 'E:\Learning\person_predict\datasets\pose_hmdb51'
csv_path = 'E:\Learning\person_predict\datasets\hmdb51_train_small.csv'

image_size = 32

validation_size = 0.2

batch_size = 128
nb_epoch= 3
#%%
df_train = pd.read_csv(csv_path).to_dict()

#文件夹列表
train_folder = df_train["sequence_name"].values()

#print("train_folder:\n",train_folder)

#%%两列交换
train_classes=df_train['class_name']
train_sequence_name=df_train['sequence_name']

train_newDict={}
for i,j in zip(train_classes.items(),train_sequence_name.items()):
    train_newDict[j[1]]=i[1]

#print("train_newDict:\n",train_newDict)    
    
#%%set：去重复
train_strClassSet=list(set(train_newDict.values()))
cls=len(train_strClassSet)
train_clsToNum={}
#类别到标签的映射
for i in range(0,cls):
    train_clsToNum[train_strClassSet[i]]=i

#print("train_clsToNum:\n",train_clsToNum)    

#%%读入原始图片
def get_datasets(path): 
    jpegs_imgList=[]
    jpegs_labelslist=[]  
    print('Going to read datasets list')
    for folder in train_folder:
        #print(folder)
        #读取到类别
        strClass=train_newDict[folder]
        path = os.path.join(jpeg_path, folder, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            image_list = fl
            label=np.zeros(cls)
            #onehot,在相应位置赋值
            label[train_clsToNum[strClass]]=1.0
            jpegs_imgList.append(image_list)
            jpegs_labelslist.append(label)
    return np.array(jpegs_imgList),  np.array(jpegs_labelslist)

jpegs_images_list,jpegs_labels = get_datasets(jpeg_path)
print("images_number:", len(jpegs_images_list))

pose_images_list,poes_labels = get_datasets(pose_path)
print("pose_number:", len(pose_images_list))
#%% shuffle操作
jpegs_images_list, pose_images_list, labels = shuffle(jpegs_images_list, pose_images_list, jpegs_labels)

num_jpegs = len(jpegs_images_list)

validation_size = int(validation_size*num_jpegs)
    
x1_train_list = jpegs_images_list[validation_size:]
x2_train_list = pose_images_list[validation_size:]
y_train = labels[validation_size:]
    
x1_validation_list = jpegs_images_list[:validation_size]
x2_validation_list = pose_images_list[:validation_size]
y_validation = labels[:validation_size]

#%%
def get_batch(path1, path2, label, batch_size, image_size):
    while True:
        for i in range(0, len(path1), batch_size):
            jpegs =[]
            poses =[]
            for fl1,fl2 in zip(path1[i:i+batch_size],path2[i:i+batch_size]):
                #read jpeg
                jpeg = cv2.imread(fl1)
                jpeg = cv2.resize(jpeg, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
                jpeg = jpeg.astype(np.float32)
                jpeg = np.multiply(jpeg, 1.0 / 255.0)
                jpegs.append(jpeg)
                #read pose
                pose = cv2.imread(fl2)
                pose = cv2.resize(pose, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
                pose = pose.astype(np.float32)
                pose = np.multiply(pose, 1.0 / 255.0)
                poses.append(pose)
        jpegs = np.array(jpegs)      
        poses = np.array(poses)    
        labels = label[i:i+batch_size]                      
        #最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
        yield ({'input1': jpegs, 'input2': poses}, {'output': labels})
         
#%%
def multi_input_model():

    input1 = Input(shape=(image_size,image_size,3), name='input1')
    input2 = Input(shape=(image_size,image_size,3), name='input2')
    
    x1 = MobileNetV2(include_top=True, weights=None,
               input_tensor=None, 
               input_shape=(image_size, image_size, 3),
               pooling=None)(input1)
 
    x2 = MobileNet(include_top=True, weights=None,
               input_tensor=None, 
               input_shape=(image_size, image_size, 3),
               pooling=None)(input2)    
    
 
    x = concatenate([x1, x2])
#    x = add([x1, x2])
    x = Dense(256, activation='relu', name='dense')(x)
    x = BatchNormalization(name='BN')(x)
    output = Dense(cls, activation='softmax', name='output')(x)
 
    model = Model(inputs=[input1, input2], outputs=[output])
    model.summary()
    return model
    
#%%    
model = multi_input_model()
# 保存模型图
plot_model(model, 'Multi_input_model.png', show_shapes=True)
 

early_stopping = EarlyStopping(monitor='val_acc', min_delta= 0,
                               patience=10, verbose=1, mode='auto',
                               baseline=None, restore_best_weights=False)

filepath = "weights-best"+str(-nb_epoch)+".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True,
                             mode='auto')

csv_logger = CSVLogger('train'+str(-nb_epoch)+'.csv')

callbacks_list = [early_stopping, checkpoint, csv_logger]

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(generator=get_batch(x1_train_list, x2_train_list, y_train, batch_size, image_size),
                    steps_per_epoch=int(len(x1_train_list)/batch_size),
                    epochs = nb_epoch,
                    verbose=1,
                    callbacks=callbacks_list,              
                    validation_data=get_batch(x1_validation_list, x2_validation_list, y_validation, batch_size, image_size),
                    validation_steps = int(len(x1_validation_list)/batch_size),
                    use_multiprocessing = False,
                    shuffle=True)
        
model.save('my-model'+str(-nb_epoch)+'.h5')



 