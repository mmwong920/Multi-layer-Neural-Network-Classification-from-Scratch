import copy
from neuralnet import *
from util import *
import util
import tqdm
import numpy as np


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    TODO: utilizing grad_check and set GradReq = False
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """
    batch_size = config['batch_size']
    epochs = config['epochs']
    early_stop = config['early_stop']
    patience = config['early_stop_epoch']

    results = {
        "training_loss": np.array([]),
        "training_accuracy": np.array([]),
        "validation_loss": np.array([]),
        "validation_accuracy": np.array([])
    }
    for i in tqdm.trange(epochs, desc='Multiclass Classification'):
        for X, t in util.generate_minibatches(util.shuffle((x_train, y_train)), batch_size=batch_size):
            model.forward(X, t)
            model.backward()

        loss, accuracy = model.forward(x_train, y_train)
        results['training_loss'] = np.append(results['training_loss'],
                                             loss)  # Using addition for potential implementation of k-fold CV
        results['training_accuracy'] = np.append(results['training_accuracy'], accuracy)
        loss, accuracy = model.forward(x_valid, y_valid)
        results['validation_loss'] = np.append(results['validation_loss'], loss)
        results['validation_accuracy'] = np.append(results['validation_accuracy'], accuracy)
        if config['early_stop']:
            if np.sum(np.diff(results['validation_loss'][-patience - 1:]) > 0) == patience:
                break

    return model, results


# This is the test method
def modelTest(model, X_test, y_test):
    """
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        tuple: (test loss, test accuracy)
    """
    return model.forward(X_test, y_test)