import argparse
import os

import keras.backend as K
import matplotlib
from dataset import Data
from keras.models import load_model
from keras.utils import to_categorical
from custom_metrics import categorical_cross_entropy, accuracy

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
The class is that which test the performance of a train model 
on the test images.

"""


class Evaluate(object):
    def __init__(self):
        self._model = None

    def load_model(self, model_path):
        """
        :param model_path: String, the filename or the path leading to the saved model to evaluate
        """
        self._model = load_model(
            model_path,
            custom_objects={"categorical_cross_entropy": categorical_cross_entropy,
                            "accuracy": accuracy}
        )

    def evaluate(self, images, masks):
        """
        test the performance of a model on some images
        :param images: A numpy 'ndarray' of size [m x 764 x 1055 x 3]. The test images
        :param masks: A numpy ndarray of size [m x 764 x 1055]. The labels
        :return:

        results: A numpy list of the results of evaluation.
        It has the loss and the accuracy (as a proportion)
        """
        masks = to_categorical(masks, num_classes=3)
        results = self._model.evaluate(images, masks, batch_size=15)
        return results

    def activation_maps(self, input):
        """
        get the activation maps for a set of images

        :param input:  A numpy ndarray of size [m x 764 x 1055 x 3]. The images to get activations of
        :return:
        input: a numpy array of size [m x 764 x 1055]. The activations of the output layer
        """
        for layer in self._model.layers:
            output = K.function([layer.get_input_at(0), K.learning_phase()],
                                [layer.get_output_at(0)])
            input = output([input, 0])[0]

        return input.squeeze()

    def plot_activations(self, output_dir, masks, activation_maps, original_images):
        """
        generate plots of activations maps and write them to directory

        :param output_dir: String, the directory to write the plots too
        :param masks: A numpy ndarray of size [m x 764 x 1055]. The ground-truth labels
        :param activation_maps: A numpy ndarray of size [m x 764 x 1055 x 3]. The predictions by the model
        :param original_images: A numpy ndarray of size [m x 764 x 1055 x 3]. The actual images
        """
        fig = plt.figure()

        # retrieve actual predictions from activation maps
        pred_y = np.argmax(activation_maps, axis=-1)

        if not os.path.exists(output_dir):
            os.system("mkdir {}".format(output_dir))
        elif not output_dir.endswith("/"):
            output_dir = output_dir + "/"

        # generates the appropriate weights to over-lay the original images
        for i in range(masks.shape[0]):
            mask_i = cv2.cvtColor(np.array(masks[i], dtype='uint8'), cv2.COLOR_GRAY2RGB)

            for j in range(mask_i.shape[0]):
                for k in range(mask_i.shape[1]):
                    if mask_i[j, k, 0] == 2:
                        mask_i[j, k, :] = [0, 255, 0]
                    if mask_i[j, k, 0] == 1:
                        mask_i[j, k, :] = [255, 102, 178]

            # over-lay the original images with the ground truth masks

            overlay = cv2.addWeighted(mask_i, 0.4, np.array(original_images[i], dtype='uint8'), 0.6, 0)
            cv2.imwrite("{}true_mask_{}.png".format(output_dir, i), overlay)
            plt.imshow(pred_y[i], cmap="Dark2_r")
            plt.xticks([])
            plt.yticks([])
            fig.savefig("{}pred_mask_{}.png".format(output_dir, i), dpi=300)

    def weighted_acc(self, test_images, ground_truth_masks):
        """
        A weighted accuracy function for testing the model's performance.
        It was necessary to redefine the function here instead of using the one defined in
        custom_metrics because that function uses Keras primitives. Here, everything is evaluated
        in numpy environment
        :param test_images: A numpy ndarray of size [m x 764 x 1055 x 3] of integers. The
        actually images to evaluate the model on

        :param ground_truth_masks: a numpy ndarray of size [m x 764 x 1055]  of ints in {0, 1, 2} - the ground-truth
        labels of the images
        :return:
        weighted_acc: a float, the weighted proportion of predictions which are correct
        """

        pred_y = self._model.predict(test_images, batch_size=15)
        pred_y = np.argmax(pred_y, axis=-1)

        N = ground_truth_masks.shape[0] * ground_truth_masks.shape[1] * ground_truth_masks.shape[2]
        mask_0 = np.array(np.logical_and(ground_truth_masks == 0, ground_truth_masks == pred_y), dtype='int')
        mask_1 = np.array(np.logical_and(ground_truth_masks == 1, ground_truth_masks == pred_y), dtype='int')
        mask_2 = np.array(np.logical_and(ground_truth_masks == 2, ground_truth_masks == pred_y), dtype='int')

        weights = (mask_0 * np.sum(mask_0)/N) + (mask_1 * np.sum(mask_1) /N) + (mask_2 * np.sum(mask_2) /N)
        weighted_acc = (ground_truth_masks == pred_y) * weights
        weighted_acc = np.sum(weighted_acc)/N
        return weighted_acc

    def main(self, model_path, data_path, output_dir=None, plot_activations=False):
        """
        Perform predictions and make plots

        :param model_path: String, the filename or the path to the saved trained model
        :param data_path: String, the path to the folder containing the test data
        :param output_dir: String, the name of the folder to save activation plots to.
        Only useful if plot_activations below is set to True
        :param plot_activations: boolean, indicates whether to plot activation maps after training or
        not. Default(True) plots activation maps.
        :return:
        accuracy: float, the raw accuracy of the evaluation
        weighted_acc: float, the weighted accuracy of the evaluation
        """

        test_data = Data()
        test_x, test_y, _, _ = test_data.load_data(directory=data_path, mode="test")

        print("\nTesting model on {} images\n".format(test_x.shape[0]))

        self.load_model(model_path)
        accuracy = self.evaluate(test_x, test_y)[1]

        weighted_acc = self.weighted_acc(test_x, test_y)

        if plot_activations:
            assert output_dir is not None

            activation_maps = np.empty_like(test_x)

            # get activations in batches of 15 images
            for i in range(0, test_y.shape[0], 15):
                activation_maps[i: i + 15] = self.activation_maps(test_x[i: i+15])
            self.plot_activations(output_dir, test_y, activation_maps, test_x)

        return accuracy, weighted_acc


if __name__ == "__main__":
    arg_passer = argparse.ArgumentParser()
    arg_passer.add_argument("--model_path", type=str, default="../saved_models/best_model_A.h5",
                            help="The folder containing the trained model to use in testing"
                            )
    arg_passer.add_argument("--data_path", type=str, default="../",
                            help="This directory should contain test_data.npz, the test data."
                            )
    arg_passer.add_argument("--output_dir", type=str, default="activations_plots_1/",
                            help="""The directory to saved activations to. Useful if
                                 plot_activations below is true """
                            )
    arg_passer.add_argument("--plot_activations", type=bool, default=False,
                            help="""Specifies whether to generate and save the activations
                                 of the final layer or no. If False(default),activation plots are not
                                 generated."""
                            )
    args, _ = arg_passer.parse_known_args()

    evaluator = Evaluate()
    test_acc, w_acc = evaluator.main(**args.__dict__)
    print("Test Accuracy: {:02.2f}%, Weighted Test Acc {:02.2f}%".format(test_acc * 100, w_acc * 100))
