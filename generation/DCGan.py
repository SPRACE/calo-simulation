from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import h5py

import sys

import numpy as np


import keras
print(keras.__version__)


def load_data():
    np_data = np.load('/data/jet-images/eminus_Ele-Eta0-PhiPiOver2-Energy50.npy')
    np.random.shuffle(np_data)
    training_size = int(np_data.shape[0]*0.8)
    training, testing = np_data[:training_size,:,:], np_data[training_size:,:,:]

    return training, testing

def plot_heatmap(data, name, file_name="plot.png"):
    fig, ax = plt.subplots()
    norm = LogNorm(vmin=0.0001, vmax=data.max())
    im0 = ax.imshow(data, cmap='inferno', norm=norm)
    ax.set_title(name)
    ax.set_xlabel('Displaced iPhi')
    ax.set_ylabel('Displaced iEta')
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)
    plt.savefig(file_name)

def generate_image_sample(quantity=1):
    gen_imgs = np.zeros((quantity,*img_shape))
    g = build_generator()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    noise = np.random.uniform(-1, 1, size=(1, 100))
    for i in range(quantity):
        gen_imgs[i] = g.predict(noise)[0]
    return gen_imgs

def save_imgs(epoch, batch=0):
    sample = generate_image_sample()
    plot_heatmap(np.squeeze(sample[0],axis=2),"Electron - Epoch {}".format(epoch), file_name="epoch_{}_batch_{}.png".format(epoch, batch))

def build_generator():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    return model


def build_discriminator():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=img_shape)
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def build_generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def train(epochs, batch_size=128, save_interval=50):

    X_train, _ = load_data()

    X_train = np.expand_dims(X_train, axis=3)

    d = build_discriminator()
    g = build_generator()
    d_on_g = build_generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/batch_size))
        for index in range(int(X_train.shape[0]/batch_size)):
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            image_batch = X_train[index*batch_size:(index+1)*batch_size]
            generated_images = g.predict(noise, verbose=0)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % save_interval == 0:
                save_imgs(epoch, index)
                g.save_weights('generator.h5', True)
                d.save_weights('discriminator.h5', True)


img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)


def main():
    train(epochs=20, batch_size=128, save_interval=100)


    _, X_test = load_data()

    data_mean = np.mean(X_test, axis=0)
    plot_heatmap(data_mean, "Test Data Mean", "test_data_mean.png")
    gen_imgs = generate_image_sample(X_test.shape[0])
    gen_imgs_mean = np.squeeze(np.mean(gen_imgs, axis=0),axis=2)
    gen_imgs_mean.shape
    plot_heatmap(gen_imgs_mean, "Mean Generated", "test_generated_mean.png")

if __name__ == "__main__":
    main()


