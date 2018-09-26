

'''

need to upgrade keras first (version 2.2.2) it will include MobileNetV2
'''


import keras
print (keras.__version__)
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
from keras.applications.mobilenetv2 import MobileNetV2,preprocess_input
#from keras.applications.nasnet import NASNetMobile,NASNetLarge
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,  img_to_array, load_img



#from keras.applications.xception import Xception, preprocess_input

from keras.layers import Dense, GlobalAveragePooling2D
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


# In[ ]:


'''
First, copy train_sample_total.zip
unzip the file

need to change following dircetories according to your own settign
'''


src_1= 'drive/My Drive/kaggle/ali/'
src_2= 'train_sample_total/' #'train data directory'
src_3 = 'guangdong_round1_test_a_20180916/'  # test data directory
src_save = 'drive/My Drive/kaggle/ali/manu_big_data'


# In[ ]:

train_file_list = os.listdir(src_2)


print (len(train_file_list))# length should be 2386


# save file list
if not os.path.isfile(src_save+'train_file_list.pkl'):

    with open (src_save+'train_file_list.pkl','wb') as handle:
        pickle.dump(train_file_list, handle)
    print ('file_list saved')
else:
    print ('file list exists')




def make_label_data(file_list):

    '''
    # get first 2 simplex character in all train_file_name (will be converted to file label)
    '''


    label_list = []
    for f in train_file_list:
        label_list.append(f[:2])
    print (len(label_list))



    # check how many failure categories
    uniq_label_list = list(set(label_list))
    print ('unique label list', len(uniq_label_list), uniq_label_list)


    '''
    convert failure category to label according to the instruction dict
    every category (2 charcter ) map to a label
    '''
    with open(src_1+'failure_to_label.pkl', 'rb') as handle:
        label_dict = pickle.load(handle)

        print (label_dict.keys())


    # label the image according to the defect type
    label_data = []
    for f in file_list:
        label_data.append(label_dict[f[:2]])
    label_data = np.array(label_data)
    label_df = pd.DataFrame (label_data)

    label_df.columns = ['label']



    # In[ ]:


    # Label Counts
    print (label_df['label'].value_counts())




    # In[ ]:

    '''
    label_data_0 : if label is 0 or not, if yes, means normal, if not means defect
    later will convert normal --True-- to '1',  defect --Fasle -- to '0'
    '''
    label_data_0 = (label_data ==0)
    label_data_0 = label_data_0*1
    print ('how many normal', sum(label_data_0))
    label_data_0[:10]

    return label_data, label_data_0


    # In[ ]:


train_label_data, train_label_data_0 = make_label_data(train_file_list)



with open (src_save+'train_label_data_verification.pkl','wb') as handle:
    pickle.dump(train_label_data, handle)
print ('label_data saved')









'''
start to make train data
will us MobileNetV2,
two stage making
stage 1: will use logic layer (the layer before dense(1000))
stage 2: will use
'''


def build_model_1():
    inputs = Input(shape=(1920,2560,3))
    #inputs = Input(shape=(224,224,3))

    # create the base pre-trained model
    base_model = DenseNet121(weights='imagenet',input_tensor=inputs,  include_top=True)
    #base_model = MobileNetV2(weights='imagenet',input_tensor=inputs,  include_top=True)
    #base_model =load_model('drive/My Drive/'+'MobileNetV2.h5')

    '''
    #(get the desired layer)
    '''

    model_1_out = base_model.layers[-2].output



    model_1 = Model(inputs=base_model.input, outputs=model_1_out)
    model_1.summary()



    return model_1





# In[ ]:


model_1= build_model_1()

'''
following wil start to make train data
as the memory limited (in colab), need to seperate the bacth making to 5 or 7 (or will encounter OOM issue)

Theoreticlly, it is no need to seperate two steps for making train dataself.
However, again, due to memeory limitation issue.  make train data step by step
'''

'''
making stage 1, use model_1
'''

model_1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:






'''
following will make bacth list
will return a final_list, each element in the final_list will be a list of file names with bacth size as pre-defined
feed to model_1 one batch at a time
the total length of batch should be equal to file list

'''
def make_batch (file_list,batch_size):
  final_list =[]


  floor = int(np.floor(len(file_list)/batch_size))
  print ('floor=', floor)
  for i in range(floor+1):
      if i < floor:
          final_list.append(file_list[i*batch_size:(i+1)*batch_size])
      else:
          final_list.append(file_list[i*batch_size:])


  # check whther the bach making final_llist len equal to file_list
  a =0
  for l in final_list:
    #print (len(l))
    #print (l[-10:])
    a += len(l)
  if (a - len(final_list) == 0):
    print ('total size', a, 'final_list length equal to file_list')
  else:
    print ('total size', a, 'file_list', len(file_list),'final_list length NOT equal to file_list')
    print (a - len(file_list))
    print ('warning, something wrong')

  return final_list


# In[ ]:





def make_data(file_list,model,src):

    '''
    first make bacth, then make data
    '''
    batch_size = 7
    final_list = make_batch(file_list,batch_size)
    print (len(final_list))




    '''
    make data
    '''

    s= time.time()  # count timing
    final_array = []  # the target array of trainging data
    for j,l in enumerate(final_list):
          img_list=[]
          for i,img in enumerate(l):



            img = load_img(src+img)

            if i == 0:
                print (img.size)
            '''
            in stage 1, no need to resize a smaller one.
            img = img.resize((2560//4,1920//4))

            '''

            x = img_to_array(img)

            '''
            stage 1 no need to apply background deletion pre_process on image

            my_filter = (x.max(axis=2) - x.min(axis=2)) < 50  # get mask

            for k in range(3):
                x[:,:,k] *= my_filter  # mask out the unnecessary part
            '''

            x = preprocess_input(x)


          img_list.append(x)
          #print ('img_list=', len(img_list))
          testing_data = np.array(img_list)


          '''
          make prediction, result will be (batch_szie, 1280)
          '''
          pred_temp = model.predict(testing_data)

          print('batch = ', j, 'stage 1 completed', 'shape=', pred_temp.shape)
          #np.save(src_save+'temp_'+str(j)+'.npy',pred_temp)

          #print ('bacth npy saved', j)
          e = time.time()
          print ('batch',j,'time=', round(e -s ,4))

          final_array.append(pred_temp) # append first batch result
          print ('afterbatch',j,'length=', len(final_array))

    '''
    convert the final_array list into numpy array and save
    Stage 1 train data done
    '''
    final_array = np.concatenate(final_array)

    return final_array


train_final_array_1 = make_data(train_file_list,model_1,src_2)
print ('train_final array_1', train_final_array_1.shape)
np.save(src_save+'train_data_DenseNet121_1_verification.npy',train_final_array_1)
print ('totally done, train_data__DenseNet121_1_verification.npy')





test_file_list =  os.listdir(src_3)


test_final_array_1 = make_data(test_file_list,model_1,src_3)
print ('test_final array_1', test_final_array_1.shape)
np.save(src_save+'test_data__Dense121_1_verification.npy',test_final_array_1)
print ('totally done, test_final_data_DenseNet_1_verification.npy')


'''
delet model 1, then start to prepare model_2
'''

del model_1








def build_model_2():


    inputs_2 = Input(shape=(480,640,3))
    base_model_2 = DenseNet121(weights='imagenet',input_tensor=inputs_2,  include_top=True)
    model_2_out = base_model_2.layers[134].output

    model_2 = Model(inputs=base_model_2.input, outputs=model_2_out)

    model_2.summary()

    return model_2






# In[ ]:


model_2 = build_model_2()



model_2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])





'''
make stage 2 train data
'''




def make_data(file_list,model,src):

    '''
    first make bacth, then make data
    '''
    batch_size = 7
    final_list = make_batch(file_list,batch_size)
    print (len(final_list))




    '''
    make data
    '''

    s= time.time()  # count timing
    final_array = []  # the target array of trainging data
    for j,l in enumerate(final_list):
          img_list=[]
          for i,img in enumerate(l):



            img = load_img(src+img)

            if i == 0:
                print (img.size)
            '''
            in stage 1, no need to resize a smaller one.
            img = img.resize((2560//4,1920//4))

            '''

            x = img_to_array(img)



            my_filter = (x.max(axis=2) - x.min(axis=2)) < 50  # get mask

            for k in range(3):
                x[:,:,k] *= my_filter  # mask out the unnecessary part


            x = preprocess_input(x)


          img_list.append(x)
          #print ('img_list=', len(img_list))
          testing_data = np.array(img_list)


          '''
          make prediction, result will be (batch_szie, 1280)
          '''
          pred_temp = model.predict(testing_data)

          print('batch = ', j, 'stage 1 completed', 'shape=', pred_temp.shape)
          #np.save(src_save+'temp_'+str(j)+'.npy',pred_temp)

          #print ('bacth npy saved', j)
          e = time.time()
          print ('batch',j,'time=', round(e -s ,4))

          final_array.append(pred_temp) # append first batch result
          print ('afterbatch',j,'length=', len(final_array))

    '''
    convert the final_array list into numpy array and save
    Stage 1 train data done
    '''
    final_array = np.concatenate(final_array)

    return final_array





train_final_array_2 = make_data(train_file_list, model_2,src_2)

print ('train final array_2', final_array_2.shape)
np.save(src_save+'train_data_img_preprocess_MobileNetV2_2_verification.npy',train_final_array_2)
print ('totally done, train_data_img_preprocess_MobileNetV2_2_verification.npy.npy')


'''
make stage 2 test data
'''



test_file_list =  os.listdir(src_3)


test_final_array_2 = make_data(test_file_list,model_2,src_3)
print ('test_final array_2', test_final_array_2.shape)
np.save(src_save+'test_data_img_preprocess_MobileNetV2_2_verification.npy',test_final_array_1)
print ('totally done, test_final_data_img_preprocess_MobileNetV2_2_verification.npy')
