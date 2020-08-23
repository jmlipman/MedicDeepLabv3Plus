from datetime import datetime
import torch
import numpy as np
from scipy import ndimage
from skimage.measure import label
import argparse

def combineLabels(inp):
    """
    In order to optimize MedicDeepLabv3+, the labels BrainMask and
    Contralateral are separated into Contralateral and Right hemisphere, 
    as the BrainMask can be calculated as the sum of the two hemispheres.
    Consequently, MedicDeepLabv3+ predictions (and the GT used to assess
    such predictions) produce Hemisphere1 and Hemisphere2 masks.
    For this reason, this function, called right before calculating the
    metrics in eval.py converts the "two hemispheres" segmentation into
    BrainMask+Contra mask.

    Args:
        `inp` (np.array): Numpy array of the predictions/ground truth
         Its shape is: BCDHW (e.g., 1,3,18,256,256).
         Channel 0: Background; Channel 1: contra; Channel 2: right hemisphere

    Returns:
        np.array after merging the labels appropriately.
    """
    labels = np.argmax(inp[0], axis=0)
    background = 1.0*(labels==0)
    brainmask = 1.0*(labels!=0)
    contra = 1.0*(labels==1)
    labels = np.stack([background, brainmask, contra], axis=0)
    labels = np.expand_dims(labels, axis=0)
    return labels
    

def np2tensor(inp):
    r"""
    Converts a numpy array into a tensor.

    Args:
        `inp` (np.array):  Numpy array to convert.

    Returns:
        (torch.Tensor): Converted array to Pytorch Tensor.
    """

    return torch.from_numpy(inp.astype(np.float32))

def log(text, output):
    r"""
    Prints a text, appending the current timestamp.
    It gets recorded by "sacred" package.

    Args:
        `text` (str): Text to print.
    """

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now + ": " + text, flush=True)
    with open(output + "log", "a") as f:
        f.write(now + ": " + text + "\n")

def softmax2onehot(_, image=None):
    """Converts a softmax probability matrix into a onehot-encoded matrix.
       Image (np.array): CHWD
    """
    if type(_) == np.ndarray:
        image = _
    result = np.zeros_like(image)
    labels = np.argmax(image, axis=0)
    for i in range(image.shape[0]):
        result[i] = labels==i
    return result

def sigmoid2onehot(_, image=None):
    """Converts a sigmoid probability matrix into a onehot-encoded matrix.
       The difference with softmax prob. matrices is that sigmoid allows
       labels to overlap, i.e., pixels can have multiple labels.
       Image (np.array): CHWD
    """
    if type(_) == np.ndarray:
        image = _
    # If it's above the thr, then it belongs to a certain class
    thr = 0.5
    result = 1.0*(image > thr)
    return result

def he_normal(w):
    r"""
    He Normal initialization.

    Args:
        `w` (torch.Tensor): Weights.

    Returns:
        Normal distribution following He initialization.
    """

    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
    return torch.nn.init.normal_(w, 0, np.sqrt(2/fan_in))

def border_np(y):
    r"""
    Calculates the surface voxels of 2D/3D binary maps.

    Args:
        `y` (binary np.array): Binary map.

    Returns:
        (np.array) containing the surface of the binary map.
    """
    return y - ndimage.binary_erosion(y)


def removeSmallIslands(masks, thr=20):
    """Post-processing to remove small islands (isolated voxels) in every
       channel. It works for N classes and 2D/3D images. What this script does:
       - Iterates over each channel.
       - Finds the location of the holes/islands to be filled.
       - Fills those values based on the neighbor voxels.

    Args:
        `masks` (np.array): Binary masks such as in one-hot encoded matrices
         or softmax matrices. The shape must be BCHWD.
        `onehot` (function): Function to transform a softmax prob. map to 1hot

    Returns:
        (np.array) Masks without small islands. BCHWD.

    """
    #from lib.transforms import Onehot
    for m in range(masks.shape[0]):
        
        onehot_mask = softmax2onehot(masks[m])

        # Incorporates a new channel to keep track of the detected holes
        onehot_mask = np.concatenate(
                (np.zeros([1] + list(onehot_mask.shape[1:])), onehot_mask),
                axis=0)

        # Deletes each small island for every channel by setting the island's
        # values to "0", and keeps track of these islands in the new channel
        # by setting "1" in this new channel 0.
        for c in range(1, onehot_mask.shape[0]):
            labelMap = label(onehot_mask[c])
            icc = len(np.unique(labelMap))
            for i in range(icc):
                if np.sum(labelMap==i) < thr:
                    onehot_mask[c, labelMap==i] = 0
                    onehot_mask[0, labelMap==i] = 1

        # At these point, the islands/holes are "deleted" from each channel
        # but it doesn't know to which class that deleted island belongs to.

        # Iterates over the location of each removed hole to figure out
        # to which class it will belong
        labelMap = label(onehot_mask[0])
        icc = len(np.unique(labelMap))
        for i in range(1, icc):
            locs = np.where(labelMap==i)
            # Takes the coordinates of a voxel in the current deleted island
            # There are 2 or 3 coordinates depending on whether 2D/3D image.
            loc = [locs[j][0] for j in range(len(locs))]
            cont = True # continue iterating
            coor = 0
            
            # Starting from `loc`, this will iterate over every coordinate
            # to figure out to which class the island will belong to.

            # This first `while` iterates over the 2-3 possible directions:
            # left-right, up-down, (and front-bottom in 3D images).
            # However, it's quite unlikely that it needs to this more than once
            while coor < len(loc) and cont:
                # This decides whether to iterate towards one direction (left)
                # or another (right).
                if loc[coor] > 0:
                    # go towards "zero"
                    it = -1
                elif loc[coor] < labelMap.shape[coor]-1:
                    # go through the opposite direction
                    it = 1
                
                new_loc = [slice(loc[j], loc[j]+1, 1) for j in range(len(locs))]
                count = 1

                # This will make it iterate over each coordinate
                while new_loc[coor].start >= 0 and new_loc[coor].start < labelMap.shape[coor]:
                    new_loc = tuple([slice(1, onehot_mask.shape[0], 1)] + new_loc)

                    # Check this coordinate
                    if np.sum(onehot_mask[new_loc]) > 0:
                        final_class = np.argmax(onehot_mask[new_loc])
                        cont = False
                        break

                    new_loc = [slice(loc[j], loc[j]+1, 1) if j!=coor else (slice(loc[j]+it*count, loc[j]+1+it*count, 1)) for j in range(len(locs))]

                    count += 1
                coor += 1

            onehot_mask[final_class+1][labelMap==i] = 1
        # Get rid of the additional channel that I've created
        masks[m] = onehot_mask[1:]

    return masks

