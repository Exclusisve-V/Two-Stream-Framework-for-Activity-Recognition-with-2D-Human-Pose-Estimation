#coding=utf-8

import pandas as pd
import numpy as np
import cv2
import os
import glob
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.applications.densenet import DenseNet121
#from keras.applications.resnet50 import ResNet50
from keras.applications import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet

from keras import optimizers

from keras.utils.vis_utils import plot_model

date_path = 'E:\\Learning\\person_predict\\datasets\\jpegs_hmdb51'
train_path = 'E:\\Learning\\person_predict\\datasets\\hmdb51_train.csv'
test_path = 'E:\\Learning\\person_predict\\datasets\\hmdb51_test.csv'

image_size = 32
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

#print("train_clsToNum:\n",train_clsToNum)    

def get_data(folders, newDict):
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
            
train_images, train_labels = get_data(train_folder, train_newDict)
print("train_shape:", train_images.shape)

test_images, test_labels = get_data(test_folder, test_newDict)
print("test_shape:", test_images.shape)

x_train, y_train = shuffle(train_images, train_labels)
x_test, y_test = shuffle(test_images, test_labels)


model = Sequential()

model.add(ResNet50(include_top=True, weights=None,
                      input_tensor=None, input_shape=(image_size, image_size, 3),
                      pooling=None,
                      classes=cls))

model.add(Activation('softmax'))

model.summary()

plot_model(model, 'Single-input-model.png', show_shapes=True)



early_stopping = EarlyStopping(monitor='val_loss', min_delta= 0,
                               patience=5, verbose=1, mode='auto',
                               baseline=None, restore_best_weights=False)

filepath="weights-best"+str(-nb_epoch)+".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='auto')

csv_logger = CSVLogger('train'+str(-nb_epoch)+'.csv')

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


#history = model.fit(x_train, y_train,
#                    batch_size=batch_size, 
#                    epochs=nb_epoch,
#                    verbose=1, 
#                    callbacks=[early_stopping, checkpoint, csv_logger],
#                    validation_split=0.2,
#                    shuffle=True)

history = model.fit(x_train, y_train,
                    batch_size=batch_size, 
                    epochs=nb_epoch,
                    verbose=1, 
                    callbacks=[early_stopping, checkpoint, csv_logger],
                    validation_data=(x_test, y_test),
                    shuffle=True)

model.save('my-model'+str(-nb_epoch)+'.h5')

## 绘制训练 & 验证的准确率值
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='best')
#plt.savefig('Model_accuracy')
#plt.close()
#
## 绘制训练 & 验证的损失值
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='best')
#plt.savefig('Model_loss')

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


y_ture=np.argmax(y_test,axis=1)
#print(classes_true)
#y_test=y_test.max(axis=1)
classes_true=pd.Series(y_ture)
classes_true.to_csv('classes_true.csv',index = False)

#print(model.predict(x_test))		# 打印概率
classes = np.argmax(model.predict(x_test), axis=1)
classes_predict=pd.Series(classes)
classes_predict.to_csv('classes_predict.csv',index = False)

