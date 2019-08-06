import numpy as np

"""
This class encapsulate functions for saving the images as compressed files
for easy transport. It also defines functions for loading the compressed images for use

@author: Maxwell Aladago
"""


class Data(object):
    def __init__(self):
        pass

    def save(self, images, labels, mode="train"):
        """
        Saves the images and masks using numpy compressed formats.

        :param images: A numpy ndarray of size [m x 764 x 1055 x 3]. The images to save
        :param labels: A numpy 'ndarray' of size [m x 764 x 1055]. The masks of the images
        :param mode: String, a flag indicating whether to save as train(default) or test. Must be
            either train or test.
        """
        if mode != "train" and mode != "test":
            print("Invalid mode, {}. mode  must be either train or test".format(mode))
            return

        # save the data in the parent folder.
        file_name = "../{}_data_patches.npz".format(mode)
        if mode == 'train':
            np.savez_compressed(file_name, train_images=images, train_labels=labels)
        else:
            np.savez_compressed(file_name,  test_images=images, test_labels=labels)

    def load_data(self, directory="./", mode='train', val_ratio=0.0):
        """
        :param directory: String, the directory containing the compressed data to load.
        This directory should have either 'train_data.npz' or 'test_data.zip'
        :param mode: String, the mode to load data for. Can either be 'train' or 'test'
        :param val_ratio: float, if mode is train, the ratio of the data to use for validation

        :return:

        train_images: a numpy ndaryy of size [m x 764 x 1055x 3] the training data.
        May also be the test data depending on mode
        train_masks: a numpy ndaryy of size [m x 764 x 1055], the labels of the training data
        val_images: a numpy ndaryy of size [r x 764 x 1055 x 3], the validation images. Only
        has data if mode is train and validation ratio is greater than 0

        """
        if mode != "train" and mode != "test":
            print("Invalid mode, {}. mode  must be either train or test".format(mode))
            return

        if not directory.endswith("/"):
            directory = directory + "/"

        file_name = "{}{}_data_patches.npz".format(directory, mode)
        data = np.load(file_name)

        images = data["{}_images".format(mode)]
        masks = data["{}_labels".format(mode)]

        if mode == 'train' and 0 < val_ratio < 1:
            num_images = len(images)
            indices = np.arange(num_images)
            np.random.shuffle(indices)

            valset_size = int(num_images * val_ratio)
            val_indices = indices[0:valset_size]
            train_indices = indices[valset_size:]

            train_images = images[train_indices]
            train_masks = masks[train_indices]
            val_images = images[val_indices]
            val_masks = masks[val_indices]
        else:
            train_images = images
            train_masks = masks
            val_images = np.array([])
            val_masks = np.array([])

        return train_images, train_masks, val_images, val_masks


def save_masks(data_path, save_masks, mode):
    """
    External API to the class above to save images
    :param data_path:
    :param mode:
    :return:
    """
    import masks
    mask_obj = masks.ImageMasks()
    data_obj = Data()
    images, labels = mask_obj.generate_masks(data_path, save_masks)
    data_obj.save(images, labels, mode)


if __name__ == "__main__":

    import argparse

    arg_passer = argparse.ArgumentParser()
    arg_passer.add_argument("--data_path", type=str,
                            default="../train_data/",
                            help="""
                            The path to the directory containing the data to generate masks.
                            The directory must contain two other sub-folders:
                            1.images
                            2. annotations
                            """
                            )
    arg_passer.add_argument("--save_masks", type=str, default=False,
                            help="A flag indicating whether to save masks or not"
                            )
    arg_passer.add_argument("--mode", type=str, default="train",
                            help="Specifies whether to save as training data or test data"
                            )

    valid_args, _ = arg_passer.parse_known_args()
    save_masks(**valid_args.__dict__)
