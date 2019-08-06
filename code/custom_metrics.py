import keras.backend as K
import tensorflow as tf


def categorical_cross_entropy(y_true, y_pred):
    """
    This function computes a weighted categorical cross entropy.
    This is essential to capture the fact imbalance in the training set
    given that there is much more background than parasites.
    :param y_true: The ground-truth one-hot labels
    :param y_pred: The predicted labels
    :return:

    Loss: The weighted loss computed for these values.
    """
    # get predicted values from softmax output
    targets = K.argmax(y_true, axis=-1)

    # compute the separate class masks using the advantage
    # of element wise boolean operations and convert
    # to floats. All trues will be converted to true.
    # Everything else will be zero
    class_0 = K.cast(K.equal(targets, 0), dtype='float32')
    class_1 = K.cast(K.equal(targets, 1), dtype='float32')
    class_2 = K.cast(K.equal(targets, 2), dtype='float32')

    # compute normalized class frequencies.
    # add a epsilon 1.e_8 to avoid numerical problems

    # use size of only one mask for normalization.
    # constant less than the total number of bits
    # because batch-size is not included but since the constant is
    # only a scaling factor, it doesn't matter

    num_class_0 = K.sum(class_0)
    num_class_1 = K.sum(class_1)
    num_class_2 = K.sum(class_2)

    # normalization constant and epsilon
    N = num_class_0 + num_class_1 + num_class_2
    e = K.epsilon()
    # normalize the values
    # compute the full weight mask by adding all the normalize class counts
    weights = N * class_0 / num_class_0
    weights = weights + (N * class_1 / (num_class_1 + e))
    weights = weights + (N * class_2 / (num_class_2 + e))

    loss = K.categorical_crossentropy(y_true, y_pred) * weights

    return loss


def accuracy(targets, pred_y):
    targets = K.argmax(targets, axis=-1)
    pred_y = K.argmax(pred_y, axis=-1)

    # get class counts
    class_0 = K.cast(K.equal(targets, 0), dtype='float32')
    class_1 = K.cast(K.equal(targets, 1), dtype='float32')
    class_2 = K.cast(K.equal(targets, 2), dtype='float32')

    num_class_0 = K.sum(class_0)
    num_class_1 = K.sum(class_1)
    num_class_2 = K.sum(class_2)

    # normalization constant and epsilon
    N = num_class_0 + num_class_1 + num_class_2
    # normalize the values
    # compute the full weight mask by adding all the individual class values
    # divided by their normalized frequencies
    weights = num_class_0 * class_0 / N
    weights = weights + (num_class_1 * class_1 / N)
    weights = weights + (num_class_2 * class_2 / N)

    acc = K.cast(K.equal(targets, pred_y), dtype='float32') * weights

    acc = K.sum(acc)/N

    return acc


def dice_coefficient(y_true, y_pred):
    targets = K.argmax(y_true, axis=-1)
    pred_y = K.argmax(y_pred, axis=-1)
    overlap = K.sum(K.cast(K.equal(targets, pred_y), dtype='int32'))
    return overlap/tf.size(targets)

