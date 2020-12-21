from __future__ import print_function

import pandas as pd
import numpy as np
import os
import sys

#sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#keras
import keras
import tensorflow as tf
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe



def data():
    feature_dir = '../input/iemocap-audio-features/'
    feature_dataframe_train = []
    for sess in range(1,5) :
        feature_df = pd.read_csv('{}audio_features_{}.csv'.format(feature_dir,sess))
        feature_df = feature_df.sample(frac=1,random_state = 50).reset_index(drop = True)
        feature_dataframe_train.append(feature_df)
    feature_dataframe_train = (pd.concat(feature_dataframe_train)).fillna(0)
    feature_dataframe_train = feature_dataframe_train.replace('exc','hap')
    
    feature_dataframe_test = []
    for sess in [5] :
        feature_df = pd.read_csv('{}audio_features_{}.csv'.format(feature_dir,sess))
        feature_df = feature_df.sample(frac=1,random_state = 50).reset_index(drop = True)
        feature_dataframe_test.append(feature_df)
    feature_dataframe_test = (pd.concat(feature_dataframe_test)).fillna(0)
    feature_dataframe_test = feature_dataframe_test.replace('exc','hap')
    feature_dataframe_train['fold'] = -1
    feature_dataframe_test['fold'] = 1
    result = (pd.concat([feature_dataframe_train, feature_dataframe_test], ignore_index=True, sort=False)).fillna(0.0)
    result = result.loc[:, (result==0.0).mean() < .9]
    training_data = result[result['fold'] == -1]
    testing_data = result[result['fold'] == 1]
    x_train_data = training_data.drop(['emotions', 'fold'],axis=1)
    x_test_data = testing_data.drop(['emotions', 'fold'],axis=1)
    y_train_data = training_data.emotions
    y_test_data = testing_data.emotions
    scaler = StandardScaler()
    std_scale = scaler.fit(x_train_data)
    x_train = std_scale.transform(x_train_data)
    x_test  = std_scale.transform(x_test_data)
    y_train_ = np.array(y_train_data)
    y_test_ = np.array(y_test_data)
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train_))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test_))
    x_train = np.expand_dims(x_train, axis = 2)
    x_test = np.expand_dims(x_test, axis = 2)
    return x_train, y_train, x_test, y_test

def model(x_train, y_train, x_test, y_test):
    
    model = Sequential()
    model.add(Conv1D(filters ={{choice([32,48,64,96,128,256,512])}},
                      padding = 'same',
                      kernel_size = {{choice([2,4,8])}},
                      input_shape = (x_train.shape[1],1)))
    model.add(Activation('relu'))
    
    model.add(Conv1D(filters={{choice([32,64,128,256])}},
                    kernel_size = {{choice([2,4,8])}},
                    padding = 'same'))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(MaxPooling1D(pool_size=(8)))
    
    model.add(Conv1D(filters={{choice([32,64,128,256])}},
                    kernel_size = {{choice([2,4,8])}},
                    padding = 'same'))
    model.add(Activation('relu'))
              
    model.add(Conv1D(filters={{choice([32,64,128,256])}},
                    kernel_size = {{choice([2,4,8])}},
                    padding = 'same'))
    model.add(Activation('relu'))
              
    model.add(Conv1D(filters={{choice([32,64,128,256])}},
                    kernel_size = {{choice([2,4,8])}},
                    padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
            
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(4)) # Target class number
    model.add(Activation('softmax'))
    metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),'accuracy']
    lr = {{choice([0.01,0.001,0.0001,0.0001])}}
    momentum = {{choice([0.00,0.50,0.99])}}
    opt = keras.optimizers.SGD(learning_rate = lr, momentum = momentum, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=metrics)
    history=model.fit(x_train, y_train, batch_size={{choice([16,64,128])}}, epochs=50, validation_data=(x_test, y_test))
    validation_acc = np.amax(history.history['val_accuracy'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
    
    
if __name__ == '__main__':
              
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials()
                                          )
    x_train, y_train, x_test, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)            
