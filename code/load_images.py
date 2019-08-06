import os

import cv2
from utils import check_directory_exist

"""
This class loads the the original images from a directory and then generates the masks
"""


class Dataset(object):
    def __init__(self, directory):

        # original dimensions of images are 1500 x 2110 x 3
        self._width = 1500
        self._height = 2110

        self._images = self._read(directory)

    def _read(self, directory):
        """
        This private utility file reads the images from the directory.
        :param directory: The folder containing the images. Directory must contain only images
        without any other file.
        :return:
        : data: a python dictionary containing the reshaped images
        as values and the file name as values.
        """
        data = {}

        directory = check_directory_exist(directory)

        images = os.listdir(directory)

        for image in images:
            # The exact label of the image is located
            # at just before the last whack to the last dot.
            # e.g if image = ../images/1.jpg, only 1 is extracted as the label

            if not image.startswith("."):
                start_index = image.rfind("/") + 1
                end_index = image.rfind(".")
                label = image[start_index:end_index]
                im = cv2.imread(directory + image)
                im = cv2.resize(im, (self._height, self._width))
                data[label] = im

        return data

    def get_images(self):
        return self._images

    def get_dimensions(self):
        num_images = len(self._images)
        return num_images, self._width, self._height

