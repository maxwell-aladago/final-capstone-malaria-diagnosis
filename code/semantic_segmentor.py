from __future__ import print_function

import argparse

from dataset import Data
from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Activation
from keras.layers import BatchNormalization, UpSampling2D,ZeroPadding2D
from keras.layers import Conv2D, Conv2DTranspose
from keras.applications import VGG16
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import custom_metrics
from losses import Losses


class VGG16FineTune(object):
    def __init__(self):

        self._num_classes = 3
        self._lr = 1e-3
        self._model = Sequential()
        self._encoder()
        self._transpose_convolution()

        optimizer = Adam(lr=self._lr)
        self._model.compile(
            loss=custom_metrics.categorical_cross_entropy,
                            optimizer=optimizer, metrics=[custom_metrics.accuracy]
        )

    def _encoder(self):
        """
        This function defines the pre-trained model to use for initializing and
        convolving to lower dimensions. Besides the original pre-trained model,
        the function fine-tunes the last three layers and adds a fully convolutional
        layer at the end
        """
        fc_kernel = 1
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(764, 1055, 3))

        for layer in vgg16.layers[:-3]:
            layer.trainable = False
        self._model.add(vgg16)
        self._model.add(Conv2D(256, (6, 6), activation='relu'))
        self._model.add(Conv2D(128, (fc_kernel, fc_kernel), activation='relu'))
        self._model.add(Conv2D(128, (fc_kernel, fc_kernel), activation='relu'))

    def _transpose_convolution(self):
        """
        This function defines the upsampling component of the semantic segmentation model.
        It uses a variety of layers typically transpose convolutional layers and a couple
        2d upsampling layers
        """
        fc_kernel = 1
        kernel = 3
        stride = 2
        self._model.add(Conv2DTranspose(128, (fc_kernel, fc_kernel), activation='relu'))
        # self._model.add(Conv2D(128, (fc_kernel, fc_kernel), activation='relu'))
        self._model.add(Conv2DTranspose(128, (fc_kernel, fc_kernel),
                                        activation='relu'))
        self._model.add(Conv2DTranspose(256, (6, 6),
                                        activation='relu'))
        self._model.add(Conv2DTranspose(64, (kernel, kernel),
                                        strides=(stride, stride),
                                        activation='relu', kernel_regularizer=rl.l2()))
        self._model.add(BatchNormalization())
        self._model.add(Conv2DTranspose(64, (kernel, kernel),
                                        strides=(stride, stride),
                                        activation='relu'))
        self._model.add(Conv2DTranspose(32, (kernel, kernel),
                                        strides=(stride, stride),
                                        activation='relu', kernel_regularizer=rl.l2()))
        self._model.add(Conv2DTranspose(8, (kernel, kernel),
                                        strides=(stride, stride),
                                        activation='relu'))

        # self._model.add(BatchNormalization())
        self._model.add(UpSampling2D())
        self._model.add(ZeroPadding2D(padding=((0, 0), (0, 3))))
        self._model.add(Conv2D(self._num_classes, (kernel, kernel)))
        self._model.add(Activation('softmax'))

    def train_model(self,
                    tr_images,
                    tr_labels,
                    val_images=None,
                    val_labels=None,
                    batch_size=15,
                    val_batch_size=2,
                    epochs=100):
        """
        Training the model

        :param tr_images: A numpy ndarray of shape [m x 764 x 1055 x 3 ] of integers. The training
        data
        :param tr_labels: A numpy ndarray of shape [m x 764 x 1055 x 3 ] of integers in {0, 1, 2}. The
        mask which are the labels for the images
        :param val_images: A numpy ndarray of shape [r x 764 x 1055 x 3 ] of integers. The images to use
        for validation
        :param val_labels: A numpy ndarray of shape [r x 764 x 1055 x 3 ] of integers in {0, 1, 2}.
        The masks of the validation data
        :param batch_size: Integer, a number specifying the number of images to use for training in
        a mini-batch
        :param val_batch_size: Integer, a number specifying the number of images to validation in each
        mini-batch
        :param epochs: Integer, specifies the number of iterations to train the model on the given data

        """

        # define generators and call backs. generators are good for managing the memory
        # requirements of the model during training

        callbacks = self._all_callbacks()
        train_gen = ImageDataGenerator(
            vertical_flip=True,
            horizontal_flip=True,
            rotation_range=90,
            data_format='channels_last'
        )

        tr_labels = to_categorical(tr_labels, num_classes=self._num_classes)
        tr_flw = train_gen.flow(tr_images, tr_labels, batch_size=batch_size)
        steps_per_epoch = int(tr_images.shape[0] / batch_size)

        # if validation data is set
        if len(val_images) > 0 and len(val_labels) > 0:
            val_gen = ImageDataGenerator(
                vertical_flip=True,
                horizontal_flip=True,
                rotation_range=90
            )

            val_labels = to_categorical(val_labels, num_classes=self._num_classes)
            val_flw = val_gen.flow(val_images, val_labels, batch_size=val_batch_size)
            val_steps = int(val_images.shape[0] / val_batch_size)
            self._model.fit_generator(
                tr_flw,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                shuffle=True,
                verbose=1,
                epochs=epochs,
                validation_data=val_flw,
                validation_steps=val_steps
            )

        # if no validation data just train
        else:
            self._model.fit_generator(tr_flw,
                                      steps_per_epoch=steps_per_epoch,
                                      callbacks=callbacks,
                                      shuffle=True,
                                      verbose=1,
                                      epochs=epochs,
                                      )

    def _predict(self, images):
        """
        Perform predictions on a given set of images

        :param images: A numpy ndarray of size [m x 764 x 1055 x 3], the images to predict masks
        :return:
        predictions: A ndarray of size [m x 764 x 1055 x 3], the predicted masks of the images.
        Note that predictions are just the activations of the output layer, apply argmax on the
        last axis (3) to get actual masks
        """
        predictions = self._model.predict(images, batch_size=2)
        return predictions

    def _all_callbacks(self, output_path='best_model_A.h5'):
        """
        This utility function bundles all the desired parameters to monitor during training
        :param output_path: the file name to use for incremental saving of the best model

        :return:
        callbacks: A python list of the set of desired call backs
        """
        checkpoint = ModelCheckpoint(
            filepath=output_path,
            monitor='accuracy', save_best_only=True
        )

        losses = Losses()
        red = ReduceLROnPlateau(verbose=1, patience=5, cooldown=1e-7)
        callbacks = [checkpoint, losses, red]
        return callbacks

    def train(self,
              directory,
              epochs,
              batch_size,
              val_ratio=0.1,
              val_batch_size=2,
              lr=1e-4,

              ):
        """
        Training the model

        :param directory: String, the directory containing the training data
        :param epochs: Integer, the number of iterations to train the model on the complete data
        :param batch_size: Integer, the number of images to use in a mini-batch for training
        :param val_ratio: float, proportion of the entire data to use for validation
        :param val_batch_size: integer, the number of images to use for validation in each mini-batch
        :param lr: float, the learning rate

        """
        self._lr = lr

        data = Data()
        tr_data, tr_labels, val_data, val_labels = data.load_data(directory=directory,
                                                                  mode="train",
                                                                  val_ratio=val_ratio
                                                                  )

        print("Training on {} images. Validating on {} "
              "images".format(tr_data.shape[0], val_data.shape[0]))

        if val_data.shape[0] <= 0:
            self.train_model(
                tr_data,
                tr_labels,
                epochs=epochs,
                batch_size=batch_size
            )

        else:
            self.train_model(
                tr_data,
                tr_labels,
                val_images=val_data,
                val_labels=val_labels,
                epochs=epochs,
                batch_size=batch_size,
                val_batch_size=val_batch_size
            )


if __name__ == '__main__':
    arg_passer = argparse.ArgumentParser()
    arg_passer.add_argument("--directory", type=str, default="../",
                            help="The directory containing the train_data.npz, the  training data"
                            )

    arg_passer.add_argument("--epochs", type=int, default=20,
                            help="The number of complete iterations to train the model"
                            )
    arg_passer.add_argument("--batch_size", type=int, default=4,
                            help="The number of images per each mini-batch"
                            )
    arg_passer.add_argument("--val_ratio", type=float, default=0.2,
                            help="The proportion of the data to use for validation"
                            )
    arg_passer.add_argument("--val_batch_size", type=int, default=2,
                            help="number of images to use for validation in each mini-batch"
                            )
    arg_passer.add_argument("--lr", type=float, default=1e-7,
                            help="The learning rate for parameter updates"
                            )

    args, _ = arg_passer.parse_known_args()
    semantic_segmentation = VGG16FineTune()
    semantic_segmentation.train(**args.__dict__)
