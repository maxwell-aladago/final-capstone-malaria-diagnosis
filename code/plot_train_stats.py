import os
from sys import argv

import matplotlib.pyplot as plt


class Plots(object):

    def __init__(self, directory):
        super().__init__()
        self._val_acc = []
        self._val_loss = []
        self._tr_acc = []
        self._tr_loss = []

        self.data(directory)

    def make_plots(self, save_prefix=""):
        """
        :param save_prefix: A string specifying a suitable prefix to save the images with.
        May be helpful in distinguishing different plots in the same folder
        """
        fig = plt.figure(figsize=(6, 5))

        plt.plot(self._tr_acc, linewidth=2, markersize=12)
        plt.plot(self._val_acc, linewidth=2, markersize=12)

        plt.ylabel("Accuracy")
        plt.xlabel("Training Epoch")
        plt.title("Training Accuracy and Validation Accuracy")
        plt.legend(["Training Accuracy", "Validation Accuracy"])

        # save this figure
        fig.savefig("{}_accuracies.png".format(save_prefix), dpi=300)

        fig = plt.figure(figsize=(6, 5))
        plt.plot(self._tr_loss, linewidth=2, markersize=12)
        plt.plot(self._val_loss, linewidth=2, markersize=12)
        plt.ylabel("Loss")
        plt.xlabel("Training Epoch")
        plt.title("Training Loss and Validation Loss")
        plt.legend(["Training Loss", "Validation Loss"])

        # save this figure
        fig.savefig("{}losses.png".format(save_prefix), dpi=300)

    def data(self, directory):
        """
        Load the necessary data for making the plots
        :param directory:  The directory containing the data
        The directory should have these four files: tr_acc.txt, val_acc.txt, tr_loss.txt and val_loss.txt
        :return:
        """
        assert os.path.isdir(directory)

        if not directory.endswith("/"):
            directory += "/"

        tr_acc_file = "{}tr_acc.txt".format(directory)
        val_acc_file = "{}val_acc.txt".format(directory)
        tr_loss_file = "{}tr_loss.txt".format(directory)
        val_loss_file = "{}val_loss.txt".format(directory)

        assert os.path.exists(tr_acc_file) and os.path.exists(val_acc_file)
        assert os.path.exists(tr_loss_file) and os.path.exists(val_loss_file)

        with open(tr_acc_file) as tr_acc, open(val_acc_file) as val_acc:
            for line in tr_acc:
                self._tr_acc = [float(x) for x in line.split(",")]

            for line in val_acc:
                self._val_acc = [float(x) for x in line.split(",")]

        with open(tr_loss_file) as tr_loss, open(val_loss_file) as val_loss:

            for line in tr_loss:
                self._tr_loss = line.split(",")

            for line in val_loss:
                self._val_loss = line.split(",")


if __name__ == "__main__":
    directory = "./"
    prefix_save = ""
    try:
        directory = argv[1]
        prefix_save = argv[2]
    except IndexError:
        pass

    plots = Plots(directory)
    print("Making plots...........................")
    plots.make_plots(prefix_save)