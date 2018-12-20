'''
Trains a simple deep NN on the MNIST dataset
using GPU allows_growth = True
Gets to ~98.40% test accuracy after 20 epochs
Aprox 3.5s per epch using NVIDIA Titan V
Original source code:
https://github.com/fchollet/keras/raw/master/examples/mnist_mlp.py
'''

from __future__ import print_function

import keras
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Dense, Dropout, GRU, Bidirectional, BatchNormalization
from keras.layers import Input, Masking, TimeDistributed, LSTM, Conv1D, Reshape
from keras.optimizers import RMSprop, Adam

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras
tf.keras.backend.set_session(sess)

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
