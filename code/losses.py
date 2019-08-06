from keras.callbacks import Callback


class Losses(Callback):
    """
    This calls extends the keras call back utility.
    By extending it, the losses and training values for each step of the training
    can be monitored. Those metrics can then be plotted later
    """
    def __init__(self):
        super().__init__()

        # statistics of interest
        self._train_losses = []
        self._val_losses = []
        self._val_acc = []
        self._training_acc = []

    def on_train_begin(self, logs={}):
        """
        This method is called by keras automatically at the beginning of every training
        session
        :param logs: the training logs. Keras supplies these logs.
        :return:
        """
        # initialize all stats at the beginning of training
        self._train_losses = []
        self._val_losses = []
        self._val_acc = []
        self._training_acc = []

    def on_epoch_end(self, epoch, logs={}):
        """
        This overrides the keras on_epoch_end function. Here, I store the
        interested parameters into a list which are retrieved at the end of training

        This method is never called explicitly by the user.
        :param epoch: The epoch number
        :param logs: The training logs.
        :return:
        """
        self._train_losses.append(logs.get('loss'))
        self._val_losses.append(logs.get('val_loss'))
        self._val_acc.append(logs.get("val_accuracy"))
        self._training_acc.append(logs.get("accuracy"))

    def on_train_end(self, logs={}):
        # save all training stats at the end of training
        # save accuracies first and then save losses
        with open("tr_acc.txt", 'wt') as tr_acc, open("val_acc.txt", 'wt') as val_acc:
            tr_acc.write(", ".join([str(tr_a) for tr_a in self._training_acc]))
            val_acc.write(", ".join([str(v_a) for v_a in self._val_acc]))

        with open("tr_loss.txt", 'wt') as tr_loss, open("val_loss.txt", 'wt') as val_loss:
            tr_loss.write(", ".join([str(tr_l) for tr_l in self._train_losses]))
            val_loss.write(", ".join([str(v_l) for v_l in self._val_losses]))

    def get_training_stats(self):
        return self._train_losses, self._val_losses, self._training_acc, self._val_acc
