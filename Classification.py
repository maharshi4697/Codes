#Import Functions

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D,concatenate, BatchNormalization, Dropout, Flatten, Activation
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
import keras.models as model
import Changes_To_Be_Made
from PIL import ImageFile


#Hyper-Parameters

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir(Changes_To_Be_Made.Directory)    #Change This Directory To The Folder That Contains your train and val folder
os.mkdir("Weights")
os.mkdir("CSV")
batch_size = 64
img_width, img_height = 224, 224
epochs = 25
learn_rate = 1e-3
ngpus = 2
nclasses =  Changes_To_Be_Made.No_Of_Classes  #Change Number Of Classes According To Your Need
model_path = './'
top_weights_path = os.path.join(
    os.path.abspath(model_path), 'Weights/top_model_weights_InceptionResnetV2_1Layer.h5')
final_weights_path = os.path.join(
    os.path.abspath(model_path), 'Weights/model_weights_InceptionResnetV2_1Layer.h5')
train_data_dir = Changes_To_Be_Made.Training_Directory     #Change This Directory To The Folder That Contains your Train Images
validation_data_dir = Changes_To_Be_Made.Validation_Directory   #Change This Directory To The Folder That Contains your Val Images
Name = "InceptionResnetV2".format(int(time.time())) 



#Image-PreProcessing

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,width_shift_range=0.3,
                                   height_shift_range=0.3,rotation_range=30,shear_range=0.5,zoom_range=.7,
                                   channel_shift_range=0.3,cval=0.5,vertical_flip=True,fill_mode='nearest')
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(
    img_height, img_width), batch_size=batch_size, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(
    img_height, img_width), batch_size=batch_size, class_mode='categorical')
train_steps = train_generator.__len__()
val_steps = validation_generator.__len__()



#Base-Model

base_model = InceptionResNetV2(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nclasses, activation='softmax')(x)
pmodel = Model(base_model.input, predictions)
model = multi_gpu_model(pmodel, ngpus)
for layer in model.layers:
    layer.trainable = True   
nadam = Nadam(lr=learn_rate)
print(f'=> creating model replicas for distributed training across {ngpus} gpus <=')
model.compile(optimizer=nadam, loss='categorical_crossentropy',metrics=['accuracy'])
print('=> done building model <=')



#Tensor-Board

tensorboard = TensorBoard(
    log_dir='./logs'.format(Name), histogram_freq=0, write_graph=True, write_images=False)
callbacks_list = [ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
                  tensorboard, EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
print('=> created callback objects <=')
print('=> initializing training loop <=')
history = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=epochs,
                              validation_data=validation_generator, validation_steps=val_steps,
                              workers=8, 
                              use_multiprocessing=True, 
                              max_queue_size=500, 
                              callbacks=callbacks_list)
print('=> loading best weights <=')
model.load_weights(final_weights_path)
print('=> saving final model <=')
pmodel.save(os.path.join(os.path.abspath(model_path), 'Weights/model_InceptionResnetV2_15Layer.h5'))




#Load-Model  #Not Required #Comment Out All Lines When Required

#new_model=tf.keras.models.load_model('/home/jediyoda/Maharshi/Coffee-Table-Material/Weights/model_InceptionResnetV2_1Layer.h5')
#new_model.summary()



#Predictions
     
predictions=[]
img_path=Changes_To_Be_Made.Image_Path   #Set This To The Val Directory Path
CSV_Name=Changes_To_Be_Made.Csv_Name    #Set CSV Name To Be Generated
filenames=validation_generator.filenames
for i in filenames:
    img = tf.keras.preprocessing.image.load_img(img_path+i, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img) 
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y=new_model.predict(x)
    y_max=y.max(axis=-1)
    if y_max>0.85:
        predict=str(y.argmax(axis=-1))
        predict=predict.replace("[","")
        predict=predict.replace("]","")
        predict=int(predict)
    else:
        predict=4
    predictions.append(predict)
labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
labels[4]='None'
predicted = [labels[k] for k in predictions]
results=pd.DataFrame({"Filename":filenames,"Predictions":predicted,"Actual":filenames})
for i in labels:
    results['Actual']=results['Actual'].str.replace(labels[i]+"/","")
results['Actual']=results['Actual'].str.replace(".jpg","")
name = []
for x in results['Actual']:
    e_n=re.sub("\d+","",x)
    name.append(e_n)
results['Actual']=name  
results.to_csv("CSV/"+CSV_Name)
