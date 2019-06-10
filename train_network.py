#!/usr/bin/python3
''' Train the network using augmented MNIST without zeros, saves the model and weights '''
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from tools import recover_dataset


if __name__ == "__main__":
    import sys
    # x, y = generate_dataset(8100)
    x = recover_dataset('data1/')
    y = np.loadtxt('data1.csv', delimiter=',').astype(int).reshape(8100)
    break_point = 7000
    x_train = x[:break_point]
    x_test = x[break_point:]
    y_train = y[:break_point]
    y_test = y[break_point:]
    y_train = y_train - 1
    y_test = y_test - 1

    batch_size = 128
    num_classes = 9 # Because we have no zeros
    epochs = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    # print(x_train.shape, y_train.shape)
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
    model_name = 'models/' +  sys.argv[1]
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name + ".h5")
