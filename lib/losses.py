# This file contains all the loss functions that I've tested, including the
# proposed "Rectified normalized Region-wise map".
#
# To simplify and for clarity reasons, I've only commented those functions
# that appear in the paper. In any case, there is a lot of repetion because
# the class "BaseData" computes the "weights" (aka Region-wise map) based
# on the name of the loss function; therefore, different loss function names
# provide different weights.
import torch
import numpy as np

def _CrossEntropy(y_pred, y_true):
    """Regular Cross Entropy loss function.
       It is possible to use weights with the shape of BWHD (no channel).

       Args:
        `y_pred`: Prediction of the model.
        `y_true`: labels one-hot encoded, BCWHD.

    """
    assert (y_pred.shape == y_true.shape), "y_pred and y_true shapes differ: " + str(y_pred.shape) + ", " + str(y_true.shape)
    ce = torch.sum(y_true * torch.log(y_pred + 1e-15), axis=1)
    return -torch.mean(ce)

def _Dice(y_pred, y_true):
    """Binary Dice loss function.

       Args:
        `y_pred`: Prediction of the model.
        `y_true`: labels one-hot encoded, BCWHD.
    """
    assert (y_pred.shape == y_true.shape), "y_pred and y_true shapes differ: " + str(y_pred.shape) + ", " + str(y_true.shape)
    axis = list([i for i in range(1, len(y_true.shape))]) # for 2D/3D images
    num = 2 * torch.sum(y_pred * y_true, axis=axis)
    denom = torch.sum(y_pred + y_true, axis=axis)
    return (1 - torch.mean(num / (denom + 1e-6)))

class Loss:
    def __init__(self, loss):
        self.loss = loss

    def __call__(self, output, y_true, weights=None):
        self.output = output
        self.y_true = y_true
        self.weights = weights
        # Execute the actual loss function
        return getattr(self, self.loss)()

    def CrossEntropyLoss(self):
        return _CrossEntropy(self.output[0], self.y_true[0])

    def DiceLoss(self):
        return _Dice(self.output[0], self.y_true[0])

    def CrossEntropyDiceLoss(self):
        ce = self.CrossEntropyLoss()
        dice = self.DiceLoss()
        return ce + dice

    def CrossEntropyDiceLoss_multiple(self):
        """CRBrain project, multiple outputs.
        """
        ce1 = _CrossEntropy(self.output[0], self.y_true[0])
        ce2 = _CrossEntropy(self.output[1], self.y_true[1])
        ce3 = _CrossEntropy(self.output[2], self.y_true[2])

        dice1 = _Dice(self.output[0], self.y_true[0])
        dice2 = _Dice(self.output[1], self.y_true[1])
        dice3 = _Dice(self.output[2], self.y_true[2])

        return ce1 + ce2 + ce3 + dice1 + dice2 + dice3

