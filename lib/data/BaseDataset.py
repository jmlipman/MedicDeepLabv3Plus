from scipy.ndimage import distance_transform_edt as dist
import torch, os
import numpy as np

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        X, Y, id_, W = self._loadSubject(idx)
        return X, Y, id_, W

    def save(self, output, loc):
        raise NotImplementedError("`save` method not implemented yet")
