from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import numpy as np
from annotations import Annotations
from load_images import Dataset


class ImageMasks:
    def __init__(self):
        self._height = 764
        self._width = 1055

    def generate_masks(self, data_path, save_masks=False):
        """
        1. for each image, retrieve the annotation object for that image
        2. put the image into a list
        3. Generate a corresponding mask for the image such that the mask has zeros everywhere
            except the regions which have been annotated. The annotated regions are assigned
            the value of the ground-truth labels
        4. Add the mask to a list such that the ground-truth image and the mask are in the
            same index in their respective lists

        :param data_path: The path to the directory containing the data. The directory must contain
        two sub-folders:
        1. annotations
        2. images
        :param save_masks: a boolean. A flag indicating whether to save the masks as images or not.
            if True, masks are saved. If False(default), masks are not saved
        :return:
            images: a numpy ndarray of size [number_images, H, W, depth] containing the ground-truth
                    images
            masks: a numpy ndarray of size [number_images, H, W] containing the masks.
        """

        if not data_path    .endswith('/'):
            data_path = "{}/".format(data_path)

        annotations_path = "{}annotations/".format(data_path)
        images_path = "{}images/".format(data_path)

        annot = Annotations(annotations_path)
        data = Dataset(images_path)

        annotations = annot.get_annotations()
        imgs = data.get_images()

        # create an empty array of the size of the images
        num_images, height, width = data.get_dimensions()

        # exclude depth for masks
        downsample_masks = np.zeros((num_images,  self._height, self._width))
        downsample_images = np.zeros((num_images, self._height, self._width, 3))

        if save_masks:
            os.system("mkdir {}d_masks/".format(data_path))

        for i in range(num_images):
            try:
                annotation = annotations[i]
                downsample_images[i] = self._downsample(imgs.get(annotation.id()))
                # process all annotations for this image
                im_annotations = annotation.annotations()
                mask = np.zeros((height, width))

                for v in im_annotations.values():
                    label = v[0]
                    try:
                        y_1, x_1, y_2, x_2 = v[1]
                        mask[x_1:x_2, y_1:y_2] = np.ones((x_2 - x_1, y_2 - y_1)) * label
                    except ValueError:
                        # handle rotated images
                        # some annotations had their coordinates recorded in rotated manner. these
                        # annotations were rotated during annotation generations so
                        # distances can never be negative.
                        # as such, they follow slightly different patches.
                        x_1, y_1, x_2, y_2 = v[1]
                        mask[x_1:x_2, y_1:y_2] = np.ones((x_2 - x_1, y_2 - y_1)) * label

                downsample_masks[i] = self._downsample(mask, mode='mask')

                if save_masks:
                    filename = "{}{}{}.png".format(data_path, "d_masks/", annotation.id())
                    cv2.imwrite(filename, downsample_masks[i])

            except:
                pass

        return downsample_images, downsample_masks

    def _downsample(self, data, mode="image"):

        """
        This method down samples image from a high resolution to a low resolution.
        This reduces memory usage especially if the original images are very large.

        :param data: A numpy 'ndarray' of size (1 x 1500 x 2110). If mode=image, data has a
        fourth dimension for the color making it (1 x 1500 x 2110 x 3)
        :param mode: Whether data is an instance of ground-truth image or the masks.
            Different interpolation methods are used for the two cases. The reason is because
            it is undesirable to average the mask since they are labels.
        :return:
        """
        if mode == "image":
            downsample = cv2.resize(data, (self._width, self._height), cv2.INTER_LINEAR)
        else:
            downsample = cv2.resize(data, (self._width, self._height), cv2.INTER_NEAREST)

        return downsample


if __name__ == '__main__':
    import argparse

    arg_passer = argparse.ArgumentParser()
    arg_passer.add_argument("--data_path", type=str,
                            default="../train_data/",
                            help="""
                                The path to the directory containing the data to generate masks.
                                The directory must contain two other sub-folders:
                                1.images
                                2. annotations
                                """)
    arg_passer.add_argument("--save_masks", type=bool,
                            default=False,
                            help="""
                                A flag indicating whether to save masks or not
                                """)

    valid_args, _ = arg_passer.parse_known_args()

    # generate masks
    ex = ImageMasks()
    images, mask = ex.generate_masks(**valid_args.__dict__)
    print(images.shape)
