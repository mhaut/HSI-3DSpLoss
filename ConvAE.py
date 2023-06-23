from keras.layers import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, Dense, Flatten, Reshape, Dropout
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np


def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))
    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))
    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))
    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))
    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))
    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))
    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    exit()
    return model


def CAE3D(input_shape=(5, 5, 200, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    model.add(Conv3D(32, (3,3,5), strides=1, padding='valid', activation='relu', name='conv1', input_shape=input_shape))
    model.add(Conv3D(32, (3,3,5), strides=1, padding='valid', activation='relu', name='conv2'))
    model.add(Flatten())
    model.add(Dense(units=25, name='embedding'))
    model.add(Dense(units=32*int(input_shape[0]-4)*int(input_shape[1]-4)*int(input_shape[2]-8), activation='relu'))
    model.add(Reshape((int(input_shape[0]-4), int(input_shape[1]-4), int(input_shape[2]-8), 32)))
    model.add(Conv3DTranspose(32, (3,3,5), strides=1, padding='valid', activation='relu', name='deconv2'))
    model.add(Conv3DTranspose(input_shape[3], (3,3,5), strides=1, padding='valid', name='deconv1'))
    model.summary()
    # exit()
    return model
