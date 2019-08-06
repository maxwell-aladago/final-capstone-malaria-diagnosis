import numpy as np
import os
from utils import check_directory_exist

"""
This class reads annotations and then prepares them for masks generations

@author Maxwell Aladago
"""


class Annotations:
    def __init__(self, annotation_path):
        self._annotations = []
        self._read_all_annotations(annotation_path)

    def _read_all_annotations(self, directory):

        # ensure the folder path given is valid
        # open the files in that folder for reading.
        # consider only .txt files
        # after opening file, check the first number.
        # if the first number is zero, indicate an all negative
        # and proceed. Otherwise,
        # create an annotation object, step through the annotations and
        # add annotations for each object.

        directory = check_directory_exist(directory)

        files = os.listdir(directory)

        for file in files:
            if file.endswith(".txt"):
                self._read_im_annotations("{}{}".format(directory, file))

    def _read_im_annotations(self, filename):
        """
        This file reads only the annotations for a given image

        :param filename: String, the file name of the image with the annotations
        """
        with open(filename) as annotation_file:
            num_annotations = annotation_file.__next__().split(",")[0]
            num_annotations = int(num_annotations)

            # consider only files which have annotations.
            if num_annotations > 0:

                # each annotation file is mapped to its image using only the file names without the
                # extensions and folder information.
                start_index = filename.rfind("/") + 1
                end_index = filename.rfind(".")
                annotation = Annotation(filename[start_index:end_index], num_annotations)

                # add all the annotations recorded  in this file.
                for line in annotation_file:
                    line = line.split(",")
                    label = line[1]

                    coordinates = np.array(line[5:13], dtype='int')

                    # rename the textual labels to integer labels
                    if label.find("Gametocytes") > -1:
                        label = 2
                    elif label.find("Trophozoites") > -1 or label.find("Schizonts") > -1:
                        label = 1

                    annotation.add_annotation(label, coordinates)
                self._annotations.append(annotation)
            else:
                print("Error", filename)

    def get_annotations(self):
        """
        This functions is a public method for which returns all the
        annotations for all the image
        :return:
            self._annotations: A python list containing annotation objects.
            # Each notation object has a filename and a dictionary of
            all annotations with the class name
        """
        return self._annotations


class Annotation:
    """
    This class models the annotation of a single image.
    # Since there can be more than one annotation per image,
    it contains modules for adding annotations to it.
    All annotations are stored in a dictionary with the positions of
    the different annotations on the image as keys
    """
    def __init__(self, filename, total_annotations):
        """"
        filename: The filename of the image with the annotations
        total_annotations; The total number of annotations this image has
        """
        self._id = filename
        self._num_annotations = total_annotations

        # keeps the different annotations for the image
        self._annotations = {}

    def add_annotation(self, class_label, coordinates):
        """
        This function appends an annotation to a given particular image
        :param class_label: The label (eg. trophozoites) associated with this annotation
        :param coordinates: a 'numpy ndarray' of size=[8x1] containing the x, y coordinates
        of the 4 corners of the square bounding box
        :return:
            Add an annotation to a particular image
        """
        # there might be more than one annotations
        # for the same type of parasite but on different regions
        # of the image. Thus, only the position is guaranteed to be distinct.
        # If more than one annotation exist on exactly, the same spot, that
        # is an ambiguous case and we can confidently override the old annotation
        coordinates = self._rotate(coordinates)
        if coordinates:
            key = str(coordinates)
            self._annotations[key] = [class_label, coordinates]

    def get_number_of_annotations(self):
        """
        :return:
           self._num_annotations: Integer, the total number of annotations for this image
        """
        return self._num_annotations

    def annotations(self):
        """
        :return:
           self._annotations: A python list, the annotations for this image
        """
        return self._annotations

    def id(self):
        """

        :return:
        self._id: String, the filename of this image.
        """
        return self._id

    def _rotate(self, coordinates):
        """

        :param coordinates: type 'numpy ndarray' of size=[8x1] containing the indices of the square bounding box.
            The coordinates are as follows:
                upper_left_coordinate(x, y) = coordinates[0], coordinates[1]
                upper_right_coordinate(x, y) = coordinates[2], coordinates[3]
                lower_right_coordinate(x, y) = coordinates[4], coordinates[5]
                lower_left_coordinate(x, y) = coordinates[6], coordinates[7]
        :return:
            slice_indices: type 'list' of size = [4 x 1] containing the indices defining
            the coordinates to be used for slicing the image.
            for example, given an image in the form of matrix X of size= [m, n],
            the region representing the annotation will be:
            X[slice_indices[0]:slice_indices[2], slice_indices[1]:slice_indices[3]]
        """
        # Ignore all annotations which have the same values for all four points of the bounding box
        if coordinates[0] == coordinates[6] or coordinates[1] == coordinates[3]:
            return None

        # set all negative points to zero
        for i in range(len(coordinates)):
            if coordinates[i] < 0:
                coordinates[i] = 0

        x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = self._organized_bounding_box(coordinates)

        # upper horizontal side.
        # take lower bound of x_1, x_2 so the parasite  can be covered
        if x_1 > x_2:
            x_1 = x_2

        # left vertical side, consider lower bound on y_1, y_4
        if y_1 > y_4:
            y_1 = y_4
        # lower horizontal side, consider upper bound on x_3, x_4
        if x_3 < x_4:
            x_3 = x_4
        # right vertical side, consider upper bound of y_2, y_3
        if y_2 < y_3:
            y_2 = y_3

        # indices for slicing
        slice_indices = [x_1, y_1, x_3, y_2]

        return slice_indices

    def _organized_bounding_box(self, coordinates):
        """
        The manner in which dragon fly generates the indices can sometimes results
        vertical or horizontal flips. This function identifies those flips and reorders them
        :param coordinates: a numpy ndarray of size [8 x 1] containing the coordinates of all the four
        corners of the square bounding box

        :return:

        The indices reordered in a proper manner such that, no negative numbers result after rotating the
        bounding boxes
        x_1: type int, the x coordinate of the upper-left point
        y_1: type int, the y coordinate of the upper-left point
        x_2: type int, the x coordinate of the upper-right point
        y_2: type int, the y coordinate of the upper-right point
        x_3: type int, the x coordinate of the lower-right point
        y_3: type int, the y coordinate of the lower-right point
        x_4: type int, the x coordinate of the lower-left point
        y_4: type int, the y coordinate of the lower-left point

        """
        x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = coordinates

        # check for proper-coordinate ordering:
        tempt_x1, tempt_y1, tempt_x2, tempt_y2 = x_1, y_1, x_2, y_2

        # if the 'suppose' upper horizontal line is actually below the lower
        # horizontal line, flip the two
        if x_1 > x_4:
            x_1 = x_4
            y_1 = y_4
            x_4 = tempt_x1
            y_4 = tempt_y1

            x_2 = x_3
            y_2 = y_3
            x_3 = tempt_x2
            y_3 = tempt_y2

        # if the 'suppose' left vertical side is actually on the right of the 'supposed' right
        # vertical side, flip the two
        tempt_x2, tempt_y2, tempt_x4, tempt_y4 = x_2, y_2, x_4, y_4
        if y_1 > y_2:
            y_2 = y_1
            x_2 = x_1
            y_1 = tempt_y2
            x_1 = tempt_x2

            y_4 = y_3
            x_4 = x_3
            y_3 = tempt_y4
            x_3 = tempt_x4

        return x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4


def retrive_annotation_stats(directory):
    """
    Get relevant statistics of the annotations

    :param directory: String, the directory containing all the annotations
    :return:
    """
    files = os.listdir(directory)
    num_nagatives = 0
    total_annotations = 0
    num_trophs = 0
    num_gametocytes = 0

    for file in files:
        with open("{}{}".format(directory, file)) as annot_file:
            num_annotations = annot_file.__next__().split(",")[0]
            num_annotations = int(num_annotations)
            if num_annotations > 0:
                total_annotations += num_annotations
                for row in annot_file:
                    row = row.split(",")
                    label = row[1]

                    if label.find("Gametocytes") > -1:
                        num_gametocytes += 1
                    elif label.find("Trophozoites") > -1 or label.find("Schizonts") > -1:
                        num_trophs += 1
                    else:
                        print(file, label)
            else:
                num_nagatives += 1

    return num_nagatives, total_annotations, num_trophs, num_gametocytes


if __name__ == '__main__':
    # change directory path to read other sets of annotations
    negatives, total, trophs, gato = retrive_annotation_stats("../train_data/annotations/")
    print("Neg: {} Tot: {} Tro: {} Gam: {}".format(negatives, total, trophs, gato))
