from __future__ import print_function

import numpy as np
from keras import models, layers, optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

"""
The context-encoder class. This class it turns out wasn't fully utilized for initializations
as originally intended. Note that the class is poorly constructed. 

@author: Maxwell Aladago
"""


class Impainter:
    def __init__(self):
        self._height = 768
        self._width = 1024
        self._channels = 3
        self._impainting_height = 197
        self._impainting_width = 197

        optim = optimizers.SGD(lr=1e-5)
        self._context_encoder = self.alexnet((227, 227, self._channels))
        self._context_encoder.compile(optimizer=optim,
                                      loss='mean_squared_error', metrics=['accuracy'])

    def alexnet(self, input_shape):
        """
        Defines an impainter model using alex net architecture
        :param input_shape: The input shape of the images
        :return:

        alex: A keras sequential model, the encoder and decoder sections of the impainter
        """
        alex = models.Sequential()

        # Block 1
        alex.add(layers.Conv2D(96, kernel_size=5, strides=1, activation='relu',
                               data_format='channels_last', input_shape=input_shape))
        alex.add(layers.MaxPooling2D((3, 3), (2, 2), data_format='channels_last'))
        alex.add(layers.BatchNormalization())
        alex.add(layers.ZeroPadding2D(2))

        # layer 2
        alex.add(layers.Conv2D(128, kernel_size=3, strides=1, activation='relu'))
        alex.add(layers.MaxPooling2D((3, 3), (2, 2)))
        alex.add(layers.BatchNormalization())
        alex.add(layers.ZeroPadding2D(1))

        # block 3
        alex.add(layers.Conv2D(256, kernel_size=3, strides=1, activation='relu'))
        alex.add(layers.Conv2D(128, kernel_size=3, strides=1, activation='relu'))
        alex.add(layers.MaxPooling2D((3, 3), (2, 2)))

        # *****************Decoder Block***************
        alex.add(layers.UpSampling2D(3))
        alex.add(layers.Conv2D(128, kernel_size=3, strides=1, activation='relu'))
        alex.add(layers.UpSampling2D(3))
        alex.add(layers.Conv2D(128, kernel_size=3, strides=1, activation='relu'))
        alex.add(layers.UpSampling2D(3))
        alex.add(layers.Conv2D(64, kernel_size=5, strides=3, activation='relu'))
        alex.add(layers.ZeroPadding2D(2))
        alex.add(layers.Conv2D(32, kernel_size=11, strides=1, activation='relu'))
        alex.add(layers.Conv2D(3, kernel_size=3, strides=1, activation='softmax'))
        return alex

    def generator(self, un_impainted_images, impainted_images, batch_size):
        """
        Yields batches of images infitely for the fit method. it helps reduces
        memory consumption of the entire model
        :return:
        """

        while True:
            indices = np.random.permutation(len(impainted_images))[0:batch_size]
            yield impainted_images[indices], un_impainted_images[indices]

    def fit(self, trainX, batch_size=10, epochs=20):
        """

        :param trainX:
        :param batch_size:
        :param epochs:
        :param validation_split:
        :return:
        """
        un_impainted_images, impainted_images, = self.impaint_imgs(trainX, 15)
        checkpoint_path = "context_encoder.h5"
        checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                     monitor='val_acc',
                                     save_best_only=True
                                     )
        decrease_lr = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.2,
                                        patience=8,
                                        min_lr=0.01)

        callbacks = [checkpoint, decrease_lr]

        steps_per_epoch = len(impainted_images)//10

        self._context_encoder.fit_generator(
            self.generator(impainted_images, un_impainted_images, batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            callbacks=callbacks
        )

    def impaint_imgs(self, images, contex_size):
        """
        impaint random portions of all images. Also keep some context around
        the impainted regions to allow the model learn

        :param images: The set of images to impaint
        :param contex_size: the amount of context to add to each image
        :return:
        original: a numpy ndarray, the original versions of the impainted sections. Will serve as
        labels to the model
        impainted_regions: a numpy ndarray, the impainted versions of the original images. Will
        serve as input to the model.
        """
        number_examples = len(images)
        shift_x = self._impainting_height + 2 * contex_size
        shift_y = self._impainting_width + 2 * contex_size
        impaint_start_x = np.random.randint(0, self._height - shift_x, number_examples)
        impaint_start_y = np.random.randint(0, self._width - shift_y, number_examples)

        impaint_end_x = impaint_start_x + shift_x
        impaint_end_y = impaint_start_y + shift_y
        inner_start_x = impaint_start_x + contex_size
        inner_end_x = inner_start_x + self._impainting_height
        inner_start_y = impaint_start_y + contex_size
        inner_end_y = inner_start_y + self._impainting_width

        impainted_regions = np.zeros((number_examples,
                                      shift_x,
                                      shift_y,
                                      self._channels))

        original = np.empty_like(impainted_regions)

        for i in range(number_examples):
            x_start, x_end = impaint_start_x[i], impaint_end_x[i]
            y_start, y_end = impaint_start_y[i], impaint_end_y[i]
            x_inner_start, x_inner_end = inner_start_x[i], inner_end_x[i]
            y_inner_start, y_inner_end = inner_start_y[i], inner_end_y[i]

            img = images[i]
            original[i] = img[y_start:y_end, x_start:x_end].copy()

            # zero out the selected region. impaint
            img[y_inner_start: y_inner_end, x_inner_start:x_inner_end] = 0
            impainted_regions[i] = img[y_start:y_end, x_start:x_end]

        return original, impainted_regions
