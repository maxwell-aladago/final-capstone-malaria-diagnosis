from __future__ import print_function

import os
from sys import exit

"""
This file contains utility functions. For example enforcing that
file names parse actually exist

@author Maxwell Aladago
"""


def checkfile_exist(filename):
    """
    Enforce that files pending opening actually exists.
    May abort the program

    :param filename: 'string' The filename of the file to open
    :return:
        filename: String, a valid file name
    """

    if not os.path.isfile(filename):
        print("Error: ", filename, " does not exist")
        filename = input("Enter filename or press 0 to exit: ")
        if filename == '0':
            exit("Aborting program. Source: user")

            checkfile_exist(filename)

    return filename


def check_directory_exist(directory):
    """
    A simple utility function for ensuring directory paths are valid.
    May abort the program

    :param directory: String, the path to the directory
    :return:
        directory: String, the directory if it's valid
    """
    if not os.path.exists(directory):
        print("Error: ", directory, " does not exist")
        exit("Aborting program. Ensure directory is valid")

    if not directory.endswith('/'):
        directory = directory + '/'

    return directory