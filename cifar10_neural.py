# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:12:19 2021

@author: Tommi Kivinen 
tommi.kivinen@tuni.fi
"""

#%% imports and data preprocessing
import pickle
import numpy as np
import matplotlib.pyplot as plt
#from random import random
#import scipy.stats as stats

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle('/Users/Tommi/Documents/dataml100/cifar-10-batches-py/test_batch')
datadict_train1 = unpickle('/Users/Tommi/Documents/dataml100/cifar-10-batches-py/data_batch_1')
datadict_train2 = unpickle('/Users/Tommi/Documents/dataml100/cifar-10-batches-py/data_batch_2')
datadict_train3 = unpickle('/Users/Tommi/Documents/dataml100/cifar-10-batches-py/data_batch_3')
datadict_train4 = unpickle('/Users/Tommi/Documents/dataml100/cifar-10-batches-py/data_batch_4')
datadict_train5 = unpickle('/Users/Tommi/Documents/dataml100/cifar-10-batches-py/data_batch_5')

X_train1 = datadict_train1["data"]
X_train2 = datadict_train2["data"]
X_train3 = datadict_train3["data"]
X_train4 = datadict_train4["data"]
X_train5 = datadict_train5["data"]

Y_train1 = datadict_train1["labels"]
Y_train2 = datadict_train2["labels"]
Y_train3 = datadict_train3["labels"]
Y_train4 = datadict_train4["labels"]
Y_train5 = datadict_train5["labels"]

X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5), axis=0)
X_train = X_train.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

Y_train = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5), axis=0)

#X_train = X_train.reshape(50000,3072).astype("int")

X = datadict["data"]
Y = datadict["labels"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

#labeldict = unpickle('/Users/Tommi/Documents/dataml100/cifar-10-batches-py/batches.meta')
#label_names = labeldict["label_names"]

#%% accuracy function

def class_acc(pred, gt):
    trueval = 0
    
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            trueval = trueval+1
            
    percent = trueval/len(pred)
    
    return f"Classifying accuracy is {percent} out of 1.0"

#%% tensorflow and keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
import keras

#%% additional preprocessing

X_train = X_train / 255
X = X / 255

# labels to one hot vectors
Y_train_1h = np.array(Y_train)
Y_train_1h = tf.one_hot(Y_train_1h.astype(np.int32), depth=10)
Y_1h = np.array(Y)
Y_1h = tf.one_hot(Y_1h.astype(np.int32), depth=10)

#%% one layer neural network model function. 
# Returns predicted value of each class for all images

def onelayer_model(n_of_neurons, lrate, n_of_epochs):
    # create model
    model = Sequential()
    
    # add layer
    model.add(Dense(n_of_neurons, input_dim=3072, activation='sigmoid'))
    
    # Output layer
    model.add(Dense(10, activation='sigmoid'))
    
    # Learning rate
    #keras.optimizers.SGD(lr=0.05)
    
    model.compile(optimizer=keras.optimizers.SGD(lr=lrate), loss='mse', metrics=['mse'])
    
    tr_hist = model.fit(X_train, Y_train_1h, epochs=n_of_epochs)
    
    plt.plot(tr_hist.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    
    test_loss, test_acc = model.evaluate(X, Y_1h, verbose=2)
    
    predictions = model.predict(X)

    return predictions

#%% testing

predictions = onelayer_model(64, 0.01, 30)
preds = [np.argmax(predictions[i]) for i in range(predictions.shape[0])]

print(class_acc(preds, Y)) # about 0.24

#%% convolutional neural network model

model2 = Sequential()

# add some layers
model2.add(Conv2D(32, 3, padding='same', input_shape=X_train.shape[1:], activation='relu'))
model2.add(Conv2D(32, 3, activation='relu'))
model2.add(MaxPooling2D())
model2.add(Dropout(0.25))

#model2.add(Conv2D(64, 3, padding='same', input_shape=X_train.shape[1:], activation='relu'))
#model2.add(Conv2D(64, 3, activation='relu'))
#model2.add(MaxPooling2D())
#model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(10, activation='softmax'))

model2.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-06), loss='categorical_crossentropy', metrics=['acc'])

#%% fitting convolutional model

tr_hist2 = model2.fit(X_train, Y_train_1h, batch_size=32, epochs=10)

# predict X with model2
predictions2 = model2.predict(X)
preds2 = [np.argmax(predictions2[i]) for i in range(predictions2.shape[0])]

# classifying accuracy
print(class_acc(preds2, Y)) # 0.68
