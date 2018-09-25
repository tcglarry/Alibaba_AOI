
'''
this program will build model for first stage and 2nd stage classification model.

Note: need to have following ready
First Stage Train data
First Stage test data
2nd Stage Train data
2nd Stage test data
First Stage traian label data
'''



import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import subprocess
import os
import pickle
import sklearn

from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBClassifier




import keras.backend as K

from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
#from keras.applications.mobilenet import MobileNet
#from keras.applications.mobilenetv2 import MobileNetV2
#from keras.applications.nasnet import NASNetMobile,NASNetLarge
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,  img_to_array, load_img



from keras.applications.xception import Xception, preprocess_input

from keras.layers import Dense, GlobalAveragePooling2D,Flatten, BatchNormalization
from keras.layers import Input, Conv2D, MaxPooling2D, merge, Lambda,UpSampling2D, concatenate, Reshape, Dropout,Cropping2D,Activation
from keras.models import Model, load_model
import pandas as pd

from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.callbacks import Callback


from keras import regularizers
#from dummyPy import OneHotEncoder
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

import itertools
import time


src_1= 'drive/My Drive/kaggle/ali/'
src_2= 'train_sample_total/' #'train data directory'
src_3 = 'guangdong_round1_test_a_20180916/'  # test data directory
src_save = 'drive/My Drive/kaggle/ali/manu_big_data/'






with open(src_1+'train_file_list.pkl', 'rb') as handle:
    file_list = pickle.load(handle)
file_list







with open(src_1+'failure_to_label.pkl', 'rb') as handle:
    label_dict = pickle.load(handle)
label_dict


'''
get first stage train data
'''

train1_data = np.load(src_save+'train_data_DenseNet121_1_verification.npy')
train1_data.shape



'''
1st stage label data if normal(which means categhory 0) , label 1
else: label 0
'''

with open (src_save+'train_label_data_verification.pkl','rb') as handle:
    train_label_data = pickle.load(handle)
train_label_data.shape

train_label_data_0 =  (train_label_data == 0) * 1


print (train_label_data_0.shape)




'''
build 1st stage model
'''

train_x,val_x,train_y,val_y = train_test_split(train1_data,train_label_data_0,test_size=0.2,shuffle=True)


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgb1.fit(train_x,train_y)


from sklearn.metrics import accuracy_score
val_pred = xgb1.predict(val_x)

predictions = [round(value) for value in val_pred]
# evaluate predictions
accuracy = accuracy_score(val_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))




'''
plot Confusion Matrix
'''











with open (src_save+'xgb1_stage_1_DenseNet121_verification.pkl','wb') as handle:
    pickle.dump(xgb1,handle)


class_names = [i for i in range (2)]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):


    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    '''
    labels = classes
    #cm = confusion_matrix(y_test, pred, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    '''
    print(cm)
    print ('\n\n')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


cnf_matrix = confusion_matrix(val_y, predictions)


# In[ ]:


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names)

from sklearn import metrics
print ('\n\n')
print (metrics.classification_report(val_y,predictions))
print ('\n\n')








'''
build 2nd stage model
# 雀屏中選
'''


train2_data = np.load(src_save+'train_data_DenseNet121_1_verification.npy')
train2_data.shape


'''
filter out defcet item from label
get  2nd stage train data and label data note: 2nd stage tarin data will use conv

'''



failure_list = train_label_data !=0
print (sum(failure_list))  # 正常: 1018, 2386-1018 = 1368
failure_list

# prepare data for 2nd stage prediction (using test_data with slicing out those to be classified as defect)
train2_data_2 = train2_data[failure_list]
print (train2_data_2.shape)



train_label_data_2 = train_label_data[failure_list]
print (train_label_data_2.shape)

'''
2nd stage label data is 1 to 11
 needs to adgest to 0 - 10 for one hot encoding
'''
one_hot_train_label_data_2 = to_categorical(train_label_data_2 - 1)

one_hot_train_label_data_2.shape




train_x,val_x,train_y,val_y = train_test_split(train2_data_2,one_hot_train_label_data_2,test_size=0.2,shuffle=True)

# NOW Data ready for training



'''
model 6 is the best result model for nd stage
'''

def conv_block(ch, strides, activation= 'relu', padding='same',kernel_regularizer=regularizers.l1(0.05)):
    return Conv2D(ch,(3,3), strides=strides, activation= activation, padding =padding , kernel_regularizer= kernel_regularizer)

def build_model_6(ch=2):
    #inputs = Input((IMAGE_HEIGHT,IMAGE_WIDTH,ch))
    inputs = Input((1024,))



    x = Dense(512, activation='relu')(inputs)

    x =Dropout(0.3)(x)

    x = Dense(256, activation='relu')(x)

    x =Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)

    x =Dropout(0.3)(x)

    #x = Dense(64, activation='relu')(x)

    #x =Dropout(0.3)(x)


    outputs = Dense(11,activation='softmax')(x)



    model = Model(inputs= inputs, outputs=outputs)
    model.summary()

    return model



model = build_model_6()

model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])

#model_6 = load_model(src_1+'conv_for_defect_11_2_final.h5')


checkpoint = ModelCheckpoint(src_save+'conv_for_defect_11_2.h5')
earlystop = EarlyStopping(monitor='val_loss',  patience=10,  mode='auto')
callback_list = [checkpoint,earlystop]

model.fit(x=train_x, y=train_y, batch_size=32, epochs=300, verbose=1, callbacks=callback_list, validation_split=0.2, shuffle=True)

model.evaluate(val_x,val_y)


model.save(src_save+'2nd_stage_dense_final_verification.h5')


'''
plot confusion matrix
'''
class_names = [i for i in range (12)]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):


    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    '''
    labels = classes
    #cm = confusion_matrix(y_test, pred, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    '''
    print(cm)
    print ('\n\n')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:

predictions = np.argmax(model.predict(val_x),axis=1)

cnf_matrix = confusion_matrix(np.argmax(val_y,axis=1), predictions)


# In[ ]:


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names)

from sklearn import metrics
print ('\n\n')
print (metrics.classification_report(np.argmax(val_y,axis=1), predictions))
print ('\n\n')




'''
***************************************stop here  ignore following*************************************





train2_data = np.load(src_1+'train_data_img_preprocess_MobileNetV2_2_verification.npy')
train2_data.shape



filter out defcet item from label
get  2nd stage train data and label data note: 2nd stage tarin data will use conv





failure_list = train_label_data !=0
print (sum(failure_list))  # 正常: 1018, 2386-1018 = 1368
failure_list

# prepare data for 2nd stage prediction (using test_data with slicing out those to be classified as defect)
train2_data_2 = train2_data[failure_list]
train_data_2.shape



train_label_data_2 = train_label_data[failure_list]
print (train_label_data_2.shape)


one_hot_train_label_data_2 = to_categorical(train_label_data_2 - 1)

one_hot_train_label_data_2.shape




train_x,val_x,train_y,val_y = train_test_split(train2_data_2,one_hot_train_label_data_2,test_size=0.2,shuffle=True)

# NOW Data ready for training




def conv_block(ch, strides, activation= 'relu', padding='same',kernel_regularizer=regularizers.l2(0.01)):
    return Conv2D(ch,(3,3), strides=strides, activation= activation, padding =padding , kernel_regularizer= kernel_regularizer)

def build_model_6(ch=8):
    #inputs = Input((IMAGE_HEIGHT,IMAGE_WIDTH,ch))
    inputs = Input((60,80,32))

    #conv0 = Conv2D(32,(2,2),padding='valid')(inputs)
    #print ('conv0',conv0.get_shape())
    conv1 =  conv_block(ch,(1,1))(inputs)
    conv1 = conv_block(ch,(1,1))(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    print ('pool1',pool1.get_shape())

    conv2 =  conv_block(ch*2,(1,1))(pool1)
    conv2 = conv_block(ch*4,(1,1))(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    print ('poo12',pool2.get_shape())


    conv3 =  conv_block(ch*8,(1,1))(pool2)
    conv3 = conv_block(ch*16,(5,5))(conv3)


    conv_embedding = Conv2D(256, (3,3),strides=(3,4), activation= 'relu', padding ='valid' )(conv3)

    flatten_x = Flatten()(conv_embedding)

    conv_embedding_2 = Dense(64)(flatten_x)

    conv_embedding_3 =Dropout(0.2)(conv_embedding_2)

    outputs = Dense(11,activation='softmax')(conv_embedding_3)



    model = Model(inputs= inputs, outputs=outputs)
    model.summary()

    return model

model = build_model_6()

model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])

#model_6 = load_model(src_1+'conv_for_defect_11_2_final.h5')


checkpoint = ModelCheckpoint(src_1+'conv_for_defect_11_2.h5')
earlystop = EarlyStopping(monitor='val_loss',  patience=10,  mode='auto')
callback_list = [checkpoint,earlystop]

model.fit(x=train_x, y=train_y, batch_size=32, epochs=300, verbose=1, callbacks=callback_list, validation_split=0.2, shuffle=True)

model.evaluate(val_x,val_y)

model.save(src_1+'conv_for_defect_11_2_final_verification.h5')
'''
