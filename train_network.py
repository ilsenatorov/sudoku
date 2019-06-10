#!/usr/bin/python3
''' Train the network using augmented MNIST without zeros, saves the model and weights '''
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import randint, choice
from os import listdir


def generate_image():
    ''' Generates a 28x28 image with a random digit 1-9, with slight degree of randomness'''
    image = Image.new(mode='L', size=(28, 28), color=0)
    txt = Image.new(mode='L', size=(28,28), color=0)
    draw = ImageDraw.Draw(txt)
    num = randint(1,9)
    fontsize = randint(26,29)
    fill = randint(0,50)
    angle = randint(-5,5)
    font = choice(listdir("fonts/"))
    font = ImageFont.truetype("fonts/" + font, size=fontsize)
    draw.text((5,1), str(num), font=font, fill=255)
    txt = txt.rotate(angle, expand=0)
    image.paste(txt)
    return np.asarray(image.getdata()), int(num)

def generate_dataset(number):
    ''' Generate a dataset of size _number_ with random images of digits '''
    x_train = np.zeros((number,784))
    y_train = np.zeros(number, int)
    for i in range(number):
        x, y = generate_image()
        x_train[i] = x
        y_train[i] = y
    return x_train.reshape(number, 28, 28)/255, y_train

if __name__ == "__main__":
    x_train, y_train = generate_dataset(40000)
    x_test, y_test = generate_dataset(10000)
    print(x_train[0])
    y_train = y_train - 1
    y_test = y_test - 1

    batch_size = 128
    num_classes = 9 # Because we have no zeros
    epochs = 2

    # input image dimensions
    img_rows, img_cols = 28, 28

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")