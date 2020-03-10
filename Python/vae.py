# Code adapted from https://github.com/lbun/VAE_Variational_Autoencoders

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import keras
from keras import models, layers
from keras import applications
import glob2 as glob
from numpy import random
from keras.datasets import mnist
import numpy as np


# dimensionality of the latents space 
embedding_dim = 32 

#Input layer
input_img = layers.Input(shape=(784,))  

#Encoding layer
encoded = layers.Dense(embedding_dim, activation='relu')(input_img)

#Decoding layer
decoded = layers.Dense(784,activation='sigmoid')(encoded) 

#Autoencoder --> in this API Model, we define the Input tensor and the output layer
#wraps the 2 layers of Encoder e Decoder
autoencoder = models.Model(input_img,decoded)
autoencoder.summary()

#Encoder
encoder = models.Model(input_img,encoded)
encoder.summary()

#Decoder
encoded_input = layers.Input(shape=(embedding_dim,))
decoder_layers = autoencoder.layers[-1]  #applying the last layer
decoder = models.Model(encoded_input,decoder_layers(encoded_input))

print(input_img)
print(encoded)

autoencoder.compile(
    optimizer='adadelta',  #backpropagation Gradient Descent
    loss='binary_crossentropy'
)

(x_train, _), (x_test, _) = mnist.load_data()  #underscore for unanimous label that we don't
                                    # want to keep im memory
#Normalization

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

x_train = x_train.reshape((-1,784))  #to go from (60000,28,28) to new shape and -1 let
                                    #numpy to calculate the number for you
x_test = x_test.reshape((-1,784))

print(x_train.shape,x_test.shape)

history = autoencoder.fit(x_train,x_train,epochs=50,batch_size=256,shuffle=True,
                validation_data=(x_test,x_test))

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()
plt.close()

# save model 
autoencoder.save('vae_autoencoder.h5')
encoder.save('vae_encoder.h5')
decoder.save('vae_decoder.h5')

# how to import model back in 
encoder = keras.models.load_model('vae_encoder.h5')
decoder = keras.models.load_model('vae_decoder.h5')

encoded_imgs = encoder.predict(x_test) 
decoded_imgs = decoder.predict(encoded_imgs)  
print(encoded_imgs.shape,decoded_imgs.shape)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape((28,28)),cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape((28,28)),cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
plt.close()

# Our written code to apply the VAE to a set of test images
def apply_vea(x_test):
    encoder = keras.models.load_model('vae_encoder.h5')
    decoder = keras.models.load_model('vae_decoder.h5')

    encoded_imgs = encoder.predict(x_test) 
    decoded_imgs = decoder.predict(encoded_imgs)

    return decoded_imgs