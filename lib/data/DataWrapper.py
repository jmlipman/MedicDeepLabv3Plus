import torch, os, random
import nibabel as nib
import numpy as np
from lib.utils import np2tensor
from lib.data.BaseDataset import BaseDataset

class DataWrapper(BaseDataset):
    """Left hemisphere + Whole brain segmentation.
       Labels overlap, so do NOT use softmax but *sigmoid*.
    """

    def __init__(self, path, split):
        """This class finds and parses the data in the provided part.
           Scans will be zero-centered and normalized to have variance of 1.
           FORMAT of the files.
           > This script uses nibabel to open images. I recommend NIfTI
             files gzip compressed i.e. scan.nii.gz.
           > Images and labels must be in the same folder.
           > All images must have the same name (variable `scanName`).
           > All labels must have the same name (variable `labelName`).
           > Labels will have values of 0s (background) and 1s (lesion).
           > Image files will have the following size:
               Height x Width x Slices x Channels. For instance: 256x256x18x1
           > Labels will have the following size:
               Height x Width x Slices. For instance: 256x256x18
           Example of `path` structure:
           `path`
            └─Study 1
              └─24h (time-point)
                ├─32 (id of the scan)
                │ ├─scan.nii.gz (image)
                │ └─scan_lesion.nii.gz (label)
                └─35
                  ├─scan.nii.gz
                  └─scan_lesion.nii.gz
           Args:
            `path`: location of the training data. It must follow the format
             described above.
            `split`: either "train" or "eval". In "eval", labels are optional.
        """

        self.split = split
        self.list = []
        self.scanName = "scan.nii.gz"
        self.brainmaskName = "scan_brainmask.nii.gz"
        self.contraName = "scan_contra.nii.gz"

        if os.path.isdir(path):
            for root, subdirs, files in os.walk(path):
                if self.scanName in files:
                    if split == "train" and (not self.brainmaskName in files or not self.contraName in files):
                        raise Exception("Parsing scans for training, but I couldn't find a the brain mask file called `"+self.brainmaskName+"` or the contra-lateral file called `"+self.contraName+"` in `"+root+"`")
                    self.list.append(root + "/")
            if len(self.list) == 0:
                raise Exception("I couldn't find any `"+self.scanName+"` file in `"+path+"`")


    def _loadSubject(self, idx):
        """This function will load a single subject.
           
           Args:
            `idx`: Index of the subject that will be read.

           Returns:
            `X`: raw brain scan.
            `Y`: labels.
            `info["id"]`: id/name of the scan.
            `W`: weights (or None)
        """

        target = self.list[idx]
        study, timepoint, subject = target.split("/")[-4:-1]
        info = {}
        info["id"] = study + "_" + timepoint + "_" + subject

        X = nib.load(target + self.scanName).get_data()
        X = np.moveaxis(X, -1, 0) # Move channels to the beginning
        X = np.moveaxis(X, -1, 1) # Move depth after channels

        X = (X - X.mean()) / X.std()

        try:
            # During training, the GT is separated into two hemispheres

            Y_brain = nib.load(target + self.brainmaskName).get_data()
            Y_contra = nib.load(target + self.contraName).get_data()
            # This is because sometimes the brainmask/contra is repeated in
            # the last dimension
            if len(Y_contra.shape) == 4:
                Y_contra = Y_contra[:,:,:,0]
            if len(Y_brain.shape) == 4:
                Y_brain = Y_brain[:,:,:,0]

            Y_brain = Y_brain.transpose(2, 0, 1)
            Y_contra = Y_contra.transpose(2, 0, 1)
            Y_both = Y_brain + Y_contra
            # I assume that Y_contra is included in Y_brain
            # So, substracting the Y_contra should display the other hemisphere
            Y_hemi = Y_brain - Y_contra
            # But, if there are voxels in Y_contra that don't belong to Y_brain
            # there will be -1s, and the following line should clean them.
            Y_hemi[Y_hemi!=1] = 0
            Y = np.stack([1.0*(Y_both==0), Y_contra, Y_hemi], axis=0)
        except FileNotFoundError:
            # Labels to assess the final masks are not provided
            Y = np.array(0)
        
        W = np.array(0) # Not used

        return [np2tensor(X)], [np2tensor(Y)], info, [np2tensor(W)]

    def save(self, output, loc):
        """Saves a mask containing both brainmask and hemisphere.
           Size of the input: 3,18,256,256
           Size of the output: 256,256,18,3
           Channel 0 = Brainmask
           Channel 1 = Hemisphere

           Args:
            `output`: numpy array with a single output. CHWD.
            `locs`: list of locations to save the data.

        """
        output = output.transpose(2, 3, 1, 0) # HWDC
        final = 1.0*(output >= 0.5) # 256,256,18,3

        nib.save(nib.Nifti1Image(final[:,:,:,1], np.eye(4)), loc + "_brainmask.nii.gz")
        nib.save(nib.Nifti1Image(final[:,:,:,2], np.eye(4)), loc + "_contra.nii.gz")
