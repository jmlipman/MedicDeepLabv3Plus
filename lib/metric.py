import numpy as np
from skimage import measure
from scipy import ndimage
from lib.utils import border_np

# Probably I can simplify the class Metric, as the only difference is the
# function executed inside.

def dice_np(pred, true):
    num = 2 * np.sum(pred * true)
    denom = np.sum(pred) + np.sum(true)
    return num / denom

def HD_np(pred, true):
    ref_border_dist, seg_border_dist = _border_distance(pred, true)
    return np.max([np.max(ref_border_dist), np.max(seg_border_dist)])

def _border_distance(pred, true):
    """Distance between two borders.
       From NiftyNet.
       y_pred and y_true are WHD
    """
    border_seg = border_np(pred)
    border_ref = border_np(true)
    distance_ref = ndimage.distance_transform_edt(1 - border_ref)
    distance_seg = ndimage.distance_transform_edt(1 - border_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg

def compactness_np(pred):
    area = np.sum(border_np(pred))
    volume = np.sum(pred)
    return area**1.5/volume

class Metric:
    def __init__(self, metrics, onehot, classes, multiprocess):
        r"""
        Initialization.

        Args:
            `metrics`: Metrics that will be measured, e.g., dice.
            `onehot`: Function to convert into a onehot encoded matrix.
             This is of special importance since softmax/sigmoid activations
             in the last layer produce mutually and non-mutually exclusive
             labels, and this needs to be considered when producing the masks
             and, consequently, when assessing them.
            `classes`: Classes that will be measured. Dict.
            `multiprocess`: Whether to use multiprocess. This is to
             make it disabled during validation.
        """
        self.metrics = metrics
        self.onehot = onehot
        self.classes = classes
        self.multiprocess = multiprocess

        if self.multiprocess:
            import multiprocessing
            self.pool = multiprocessing.Pool(processes=4)

    def all(self, y_pred, y_true):
        r"""
        Helper function. Decides whether to multiprocess or not

        Returns:
            Either the computed results or a "pooled instance".
        """
        if self.multiprocess:
            return self.pool.apply_async(self.all_, args=(y_pred, y_true))
        else:
            return self.all_(y_pred, y_true)

    def all_(self, y_pred, y_true):
        r"""
        Computes all required metrics.

        Returns:
            Dictionary with keys=metrics, values=results.
        """
        results = {}
        for metric in dir(self):
            if metric in self.metrics:
                # Given the metric name, execute it and get the result
                # Return result from each function will be a dictionary
                # of the form: {"metric_name": [result]}
                results.update(getattr(self, metric)(y_pred, y_true))

        return results

    def _getMean(self, results):
        r"""
        Gets the average of the results. If matrix `results` contains -1
        it means that such class in such sample was not found in the ground
        truth, and therefore, the current metric was not computed.

        This function calculates the mean of the computed results accounting
        for labels and cases that were not computed, which will be -1.

        Args:
            `results` (np.array): Matrix of size n_samples, n_classes

        Returns:
            (list) of size `n_classes` with the mean metric per class
            accounting for those cases where such class was found. If
            no samples had such class, the it produces a value of -1.
        """

        final_results = []
        for c in sorted(self.classes):
            elem = results[:,c][results[:,c] != -1]
            if len(elem) == 0:
                final_results.append(-1)
            else:
                final_results.append(elem.mean())

        return final_results

    def dice(self, y_pred_all, y_true_all):
        r"""
        This function calculates the Dice coefficient.
        Works for 2D and 3D images.
        Input size: BCHWD

        Returns:
           List with Dice coefficients. len = C
        """
        
        n_samples = y_pred_all.shape[0]
        # -1 to know in which classes this metric was not computed
        results = np.zeros((n_samples, len(self.classes))) - 1

        for n in range(n_samples):
            y_pred = self.onehot(y_pred_all[n])
            for c in sorted(self.classes):
                pred = 1.0*(y_pred[c] > 0.5)
                true = 1.0*(y_true_all[n, c] > 0.5)

                if np.sum(true) > 0: # If class c is in the ground truth
                    # Dice coeff. formula
                    #num = 2 * np.sum(pred * true)
                    #denom = np.sum(pred) + np.sum(true)
                    #results[n, c] = num / denom
                    results[n, c] = dice_np(pred, true)

        return {"dice": self._getMean(results)}

    def HD(self, y_pred_all, y_true_all):
        """Hausdorff distance.
           From NiftyNet.
        """

        n_samples = y_pred_all.shape[0]
        # -1 to know in which classes this metric was not computed
        results = np.zeros((n_samples, len(self.classes))) - 1

        for n in range(n_samples):
            y_pred = self.onehot(y_pred_all[n])
            for c in sorted(self.classes):
                pred = 1.0*(y_pred[c] > 0.5)
                true = 1.0*(y_true_all[n, c] > 0.5)
                if np.sum(true) > 0:
                    #ref_border_dist, seg_border_dist = self._border_distance(pred, true)
                    #results[n, c] = np.max([np.max(ref_border_dist), np.max(seg_border_dist)])
                    results[n, c] = HD_np(pred, true)

        return {"HD": self._getMean(results)}

    def compactness(self, y_pred_all, y_true_all):
        """surface^1.5 / volume
        """
        n_samples = y_pred_all.shape[0]
        results = np.zeros((n_samples, len(self.classes))) - 1

        for n in range(n_samples):
            y_pred = self.onehot(y_pred_all[n])
            for c in sorted(self.classes):
                pred = 1.0*(y_pred[c] > 0.5)
                if np.sum(pred) > 0:
                    #area = np.sum(border_np(pred))
                    #volume = np.sum(pred)
                    #results[n, c] = area**1.5/volume
                    results[n, c] = compactness_np(pred)

        return {"compactness": self._getMean(results)}


    def __getstate__(self):
        """Executed when doing.get(). If we don't delete the pool
           it will return an Exception.
           Read more here: https://stackoverflow.com/questions/25382455/python-notimplementederror-pool-objects-cannot-be-passed-between-processes/25385582
        """
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def close(self):
        """Closes the pool used for multiprocessing.
        """
        if self.multiprocess:
            self.pool.close()
            self.pool.join()
            self.pool.terminate()
