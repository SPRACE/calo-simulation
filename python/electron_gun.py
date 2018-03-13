#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   file: electron_gun.py
   Choose the energy of the electron and generate images
"""
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, Input, Lambda, LeakyReLU,
                          LocallyConnected2D, Reshape, UpSampling2D)

from keras.models import Model

from matplotlib.colors import LogNorm

from ops import scale, inpainting_attention

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def build_generator(x, nb_rows, nb_cols):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        x: a keras Input with shape (None, latent_dim)
        nb_rows: int, number of desired output rows
        nb_cols: int, number of desired output cols

    Returns:
    --------
        a keras tensor with the transformation applied
    """

    x = Dense((nb_rows + 2) * (nb_cols + 2) * 36)(x)
    x = Reshape((nb_rows + 2, nb_cols + 2, 36))(x)

    x = Conv2D(16, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = LocallyConnected2D(6, (2, 2), kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)

    x = LocallyConnected2D(1, (2, 2), use_bias=False,
                           kernel_initializer='glorot_normal')(x)

    return x


def generator_inputs(latent_size):
    latent = Input(shape=(latent_size, ), name='z')
    input_energy = Input(shape=(1, ), dtype='float32')
    inputs = [latent, input_energy]
    return inputs


def generator_outputs(inputs, sizes):
    latent = inputs[0]
    input_energy = inputs[1]
    h = Lambda(lambda x: x[0] * x[1])([latent, scale(input_energy, 100)])

    img_layer0 = build_generator(h, sizes[0], sizes[1])
    img_layer1 = build_generator(h, sizes[2], sizes[3])
    img_layer2 = build_generator(h, sizes[4], sizes[5])

    avgpool = AveragePooling2D(pool_size=(1, 8))
    zero2one = avgpool(UpSampling2D(size=(4, 1))(img_layer0))
    img_layer1 = inpainting_attention(img_layer1, zero2one)

    one2two = AveragePooling2D(pool_size=(1, 2))(img_layer1)
    img_layer2 = inpainting_attention(img_layer2, one2two)

    outputs = [Activation('relu')(img_layer0),
               Activation('relu')(img_layer1),
               Activation('relu')(img_layer2)]

    return outputs


def generate_images(batch_size, latent_size, energy, generator, fname):
    '''
    Args:
    -----
        batch_size: int, number of images to generate
        latent_size: int, dimention of the latent vector
        energy: float, energy of the particle
        generator: Keras model
        fname> str, prefix for the output images
    '''
    noise = np.random.normal(0, 1, (batch_size, latent_size))
    energy_array = np.array([[energy]], dtype=float)
    images = generator.predict([noise, energy_array])

    images = map(lambda x: np.squeeze(x * 1000), images)

    for i, img in enumerate(images):
        plot_image(img, sizes, layer=i,
                   vmin=max(img.mean(axis=0).min(), 1.e-3),
                   vmax=img.mean(axis=0).max())

        out = 'images/{}_{}_layer_{}.pdf'.format(fname, str(int(energy)), i)
        plt.savefig(out, transparent=True)


def plot_image(image, sizes, layer, vmin=None, vmax=None):
    '''
    Args:
    -----
        image: ndarray with energies collected by each calo cell
        layer: int in {0,1,2}, useful to resize image correctly
        vmin: float, min energy to clip at
        vmax: float, max energy to clip at
    '''
    fig = plt.figure(figsize=(20, 20))
    im = plt.imshow(image,
                    aspect=float(sizes[layer*2 + 1])/sizes[layer*2],
                    interpolation='nearest',
                    norm=LogNorm(vmin, vmax))
    cbar = plt.colorbar(fraction=0.0455)
    cbar.set_label(r'Energy (MeV)', y=0.83)
    cbar.ax.tick_params()

    xticks = range(sizes[layer*2 + 1])
    yticks = range(sizes[layer*2])
    if layer == 0:
        xticks = xticks[::10]
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(r'$\eta$ Cell ID')
    plt.ylabel(r'$\phi$ Cell ID')

    plt.tight_layout()
    return im


if __name__ == '__main__':
    batch_size = 1
    latent_size = 1024
    sizes = [3, 96, 12, 12, 12, 6]

    matplotlib.rcParams.update({'font.size': 50})  # Cosmetics

    inputs = generator_inputs(latent_size)
    outputs = generator_outputs(inputs, sizes)
    generator = Model(inputs, outputs)

    energy = 50  # GeV

    generate_images(batch_size, latent_size, energy, generator, 'backg')

    generator.load_weights('weights/params_generator_epoch_049.hdf5')

    generate_images(batch_size, latent_size, energy, generator, 'signal')
