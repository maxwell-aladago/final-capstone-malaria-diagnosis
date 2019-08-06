import os
from utils import check_directory_exist
import numpy as np
"""
A file to divide the data into train and test subgroups. The test was set separately util 
evaluation time. 

This not required for running the code

@author Maxwell Aladago
"""


def train_test_split(data_path, test_path, test_ratio):
    """
    This utility function separates the images and the annotations into training
    and testing groups. The files for testing are moved into a separate folder whilst
    those for testing are kept in the same folder

    :param data_path: The path containing all the data. This folder will contain only the data for
    training after this function has finished executing
    :param test_path: The path to store the test data. Will contain two subdirectories after this function
    :param test_ratio: 'float'. Strictly between 0 and 1. The ratio of the data to use for testing.
    :return:
    """
    data_path = check_directory_exist(data_path)
    assert test_path is not None
    assert test_ratio > 0
    assert test_ratio < 1

    # create the test directory if it's not already there
    if not os.path.exists(test_path):
        print("Creating directory {} ".format(test_path))
        os.system("mkdir {}".format(test_path))

    # create train and annotations directories.
    os.system("mkdir {}/{} {}/{}".format(test_path, "images/", test_path, "annotations/"))

    # prepare directory paths
    images_path = check_directory_exist(data_path + "images/")
    annotations_path = check_directory_exist(data_path + "annotations/")
    test_path_images = "{}/images".format(test_path)
    test_path_annotations = "{}annotations".format(test_path)

    images = os.listdir(images_path)
    # annotations = os.listdir(annotations_path)

    num_files = len(images)
    indices = np.arange(num_files)
    np.random.shuffle(indices)
    num_test = int(np.floor(num_files * test_ratio))
    test = indices[0:num_test]

    print("Using {} for training and {} for testing".format(num_files - num_test, num_test))

    for i in test:
        im = images[i]
        ann = "{}txt".format(im[:-3])
        os.system("mv {}{} {}".format(images_path, im, test_path_images))
        os.system("mv {}{} {}".format(annotations_path, ann, test_path_annotations))


if __name__ == "__main__":
    train_test_split("../train_data/",
                     "../test_data/",
                     0.1
                     )
