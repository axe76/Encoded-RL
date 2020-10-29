# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 00:21:15 2020

@author: ACER
"""

from keras.models import Model
from keras.layers import Dense,Reshape,Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Flatten
from keras.layers import MaxPooling2D,UpSampling2D
from keras import backend as K
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from keras.layers import LeakyReLU

x_train = image.load_img(r"test.jpg",target_size = (224,224))
x_train = image.img_to_array(x_train)
img = Image.open(r"test.jpg")
img.show()
x_train.shape
x_train = x_train.astype('float32')/255

input_shape = (x_train.shape[0],x_train.shape[1],x_train.shape[2])

inputs = Input(shape = input_shape)
x = inputs
x = Conv2D(64,(3,3),activation = LeakyReLU(alpha=0.5))(x)#'relu'
#x = MaxPooling2D(pool_size = (2,2))(x)
x = Conv2D(32,(3,3),activation = 'tanh')(x)
#x = MaxPooling2D(pool_size = (2,2))(x)
shape = K.int_shape(x)
x = Flatten()(x)
latent = Dense(units = 32, activation = 'tanh')(x)#'sigmoid'

encoder = Model(inputs,latent)
encoder.summary()

latent_input = Input(shape = (32,))
x = Dense(shape[1]*shape[2]*shape[3],activation = 'tanh')(latent_input)#'sigmoid'
x = Reshape((shape[1],shape[2],shape[3]))(x)
#x = UpSampling2D(size = (2,2))(x)
x = Conv2DTranspose(64,(3,3),activation = 'tanh')(x)
#x = UpSampling2D(size = (2,2))(x)
#x = Conv2DTranspose(3,(3,3),activation = LeakyReLU(alpha=0.5))(x)#'relu'
output = Conv2DTranspose(3,(3,3),activation = LeakyReLU(alpha=0.5))(x)
#output = Conv2DTranspose(3,(3,3),activation = 'relu')(x)#'relu'LeakyReLU(alpha=0.5)

decoder = Model(latent_input,output)
decoder.summary()

autoencoder = Model(inputs,decoder(encoder(inputs)))
autoencoder.summary()

x_train = np.expand_dims(x_train,axis = 0)

autoencoder.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
autoencoder.fit(x_train,x_train,epochs=300, batch_size=1)

latent_vec = encoder.predict(x_train)
latent_vec
decrypted = decoder.predict(latent_vec)

op = autoencoder.predict(x_train)
op = np.squeeze(op,axis = 0)
op_img = (op*255).astype(np.uint8)

op_img = image.array_to_img(op_img)
op_img.show()

ip = np.squeeze(x_train,axis = 0)
ip_img = (ip*255).astype(np.uint8)
ip_img = image.array_to_img(ip_img)
ip_img.show()
#img = Image.fromarray(op)

#img = Image.fromarray()




