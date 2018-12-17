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

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2  # Latent space dimensions
validation_fraction = 0.2
energyCut = 35.0  # GeV

# signals used to train and test the model
name = "eminus_Ele-Eta0-PhiPiOver2-Energy50_28x28.npy"
data = np.load(name)
print(data.shape)

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

#fig = plot_hist(mean_signal)

energyArray = sum_energy(data)

#plot_energy_hist(energyArray, 100)

print("Average Energy: " + str(np.mean(energyArray)))
print("Maximum Energy: " + str(np.max(energyArray)))
print("Minimum Energy: " + str(np.min(energyArray)))

data_t = data[energyArray > energyCut]
data_t.shape

#plt.title("All signals energy (max = 50 GeV) - sorted low to high")
#plt.xlabel("Signal: 0 to " + str(data_t.shape[0]))
#plt.ylabel("energy (i-th signal) [GeV]")
#plt.grid()
#plt.semilogy()
#plt.plot((np.sort(energyArray)))
#plt.show()

signal = energyArray

numSelectedEvents = data_t.shape[0]
print(numSelectedEvents)
signal_t = np.zeros(numSelectedEvents)
for i in range(numSelectedEvents):
    signal_t[i] = sum_energy(data_t[i])

#plt.title("Selected signals energy (max = 50 GeV) - sorted low to high")
#plt.xlabel("Sinal: 0 a " + str(data_t.shape[0]))
#plt.ylabel("energy (i-th signal) [GeV]")
#plt.grid()
#plt.plot((np.sort(signal_t)))
#plt.show()

# # More validation

#mean_selected_signal = np.mean(data_t, axis=0)
#fig = plot_hist(mean_selected_signal)

# # Model definition

# Using only the selected signal from now on
numSelectedEvents = data_t.shape[0]

# ## Notice that the model itself is still a black box for us...

input_img = keras.Input(shape=(img_shape))

### Base CNN
x = layers.Conv2D(32, 3, padding="same", activation="relu")(input_img)
x = layers.Conv2D(64, 3, padding="same", activation="relu", strides=(2, 2))(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

### AddOn
x = layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="same", data_format=None
)(x)
x = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest")(x)
# x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
# x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

### Dense NN
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.
    )
    return z_mean + K.exp(z_log_var) * epsilon


z = layers.Lambda(sampling)([z_mean, z_log_var])

decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation="relu")(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(32, 3, padding="same", activation="relu", strides=(2, 2))(x)
x = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
decoder = Model(decoder_input, x)
z_decoded = decoder(z)


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

data_t.shape

data_t_train.shape

data_t_test.shape

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
    epochs=20,
    batch_size=100,
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

# In[ ]:

#plot_hist(partial_signals[np.random.randint(0, partial_signals.shape[0])])

# In[ ]:

mean_synthetic_signal = np.mean(partial_signals, axis=0)
mean_synthetic_signal.shape

# In[ ]:

#plot_hist(mean_synthetic_signal)

# In[ ]:

syntheticEnergyArray = sum_energy(partial_signals)
print("Average Energy: " + str(np.mean(syntheticEnergyArray)))
print("Maximum Energy: " + str(np.max(syntheticEnergyArray)))
print("Minimum Energy: " + str(np.min(syntheticEnergyArray)))
#plot_energy_hist(syntheticEnergyArray, 100)

# In[ ]:

#plot_cumulative(data=mean_eta(data_t), ylabel="Energy [GEV]", xlabel="Eta")

# In[ ]:

#plot_cumulative(data=mean_eta(partial_signals), ylabel="Energy [GEV]", xlabel="Eta")

np.save("syntethic_signals.npy", partial_signals, allow_pickle=False)
