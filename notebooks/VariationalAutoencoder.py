# coding: utf-8

import sys

sys.path.append("../python/lib/")

import keras
from keras import layers
from keras import backend as K
from keras import optimizers
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

### Technical cuts
img_shape = (28, 28, 1)
batch_size = 100
original_dim = img_shape[0]*img_shape[1]
latent_dim = 3  # Latent space dimensions
intermediate_dim = 512
nb_epoch = 10
validation_fraction = 0.2

### Physics cuts
energyCut = 35.0  # GeV

# signals used to train and test the model
name = "eminus_Ele-Eta0-PhiPiOver2-Energy50_28x28.npy"
data = np.load(name)
print(data.shape)
print(original_dim)

# importing generate function from lib util
from util import generate
from util import sum_energy
from util import mean_eta

# importing plot_hist function from histograms
from histograms import plot_hist
from histograms import plot_energy_hist
from histograms import plot_cumulative

# # Validation

mean_signal = np.mean(data, axis=0)
energyArray = sum_energy(data)

print("Average Energy: " + str(np.mean(energyArray)))
print("Maximum Energy: " + str(np.max(energyArray)))
print("Minimum Energy: " + str(np.min(energyArray)))

# # Select events with energy above energy cut
data_t = data[energyArray > energyCut]
print("Selected events shape",data_t.shape)
signal = energyArray

numSelectedEvents = data_t.shape[0]
signal_t = np.zeros(numSelectedEvents)
for i in range(numSelectedEvents):
    signal_t[i] = sum_energy(data_t[i])

# ## Notice that the model itself is still a black box for us...

### Flatten
input_img = keras.Input(shape=(img_shape))
shape_before_flattening = K.int_shape(input_img)
x = layers.Flatten()(input_img)

### DNN for encoding
x = layers.Dense(intermediate_dim, activation="relu")(x)
x = layers.Dense(intermediate_dim, activation="relu")(x)
x = layers.Dense(intermediate_dim, activation="relu")(x)
x = layers.Dense(intermediate_dim, activation="relu")(x)
#x = layers.Dense(intermediate_dim, activation="relu")(x)
#x = layers.Dense(intermediate_dim, activation="relu")(x)
#x = layers.Dense(intermediate_dim, activation="relu")(x)
#x = layers.Dense(intermediate_dim, activation="relu")(x)

### Encoder
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
print("z_mean shape",z_mean)
print("z_log_var shape",z_log_var)

'''
Define a sampling function.
The decoder takes z as its input
and output the parameters to the probability distribution of the data.
Epsilon is a random normal tensor.
'''
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.
    )
    return z_mean + K.exp(z_log_var) * epsilon
z = layers.Lambda(sampling)([z_mean, z_log_var])
print("z shape",z)

### Decoder
decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation="relu")(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
decoder = Model(decoder_input, x)
z_decoded = decoder(z)
print("z_decoded shape",z_decoded)

### This part I still don't understand...
class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
        )
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x
y = CustomVariationalLayer()([input_img, z_decoded])
print("y shape",y)

# ## Compile model

vae = Model(input_img, y)
vae.compile(optimizer="rmsprop", loss=None)

# ## Separate training and validation datasets

print(numSelectedEvents)
numTrainEvents = int(np.round((1.0 - validation_fraction) * numSelectedEvents))
print(
    "20% validation: "
    + str(numTrainEvents)
    + "/"
    + str(numSelectedEvents)
    + " for training"
)
data_t_train = data_t[:numTrainEvents, :, :]
data_t_test = data_t[numTrainEvents:, :, :]

# ## Final checks

print("data_t.shape",data_t.shape)
print("data_t_train.shape",data_t_train.shape)
print("data_t_test",data_t_test.shape)

# ## Fit - this takes time
# ### 80 minutes in Thiago's laptop

data_t_train = data_t_train.reshape(data_t_train.shape + (1,))
print(data_t_train.shape)
data_t_test = data_t_test.reshape(data_t_test.shape + (1,))
print(data_t_test.shape)
history = vae.fit(
    x=data_t_train,
    y=None,
    shuffle=True,
    epochs=nb_epoch,
    batch_size=batch_size,
    validation_data=(data_t_test, None),
)

# Build synthetic signals from the latent space of the autoencoder
limit_i = 145
limit_j = 145
signal_counter = 0

partial_signals = np.zeros((limit_i * limit_j, 28, 28))
print(partial_signals.shape)
for i in range(limit_i):
    for j in range(limit_j):
        if i % 10 == 0 and j % 10 == 0:
            print(i, j)
        z_sample = np.array([[i, j]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        partial_signals[signal_counter, :, :] = x_decoded[0].reshape(28, 28)
        signal_counter = signal_counter + 1

mean_synthetic_signal = np.mean(partial_signals, axis=0)
syntheticEnergyArray = sum_energy(partial_signals)
print("Average Energy: " + str(np.mean(syntheticEnergyArray)))
print("Maximum Energy: " + str(np.max(syntheticEnergyArray)))
print("Minimum Energy: " + str(np.min(syntheticEnergyArray)))

np.save("syntethic_signals.npy", partial_signals, allow_pickle=False)
