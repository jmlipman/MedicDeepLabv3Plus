import torch 
from lib.utils import log, he_normal, removeSmallIslands, combineLabels
from lib.utils import softmax2onehot, sigmoid2onehot
import os, time, json
from torch import nn
from datetime import datetime
import numpy as np
from lib.metric import Metric

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def initialize(self, device, output="", model_state=""):
        """Sets the device, output path, and loads model's parameters if needed

           Args:
            `device`: Device where the computations will be performed. "cuda:0"
            `output`: Path where the output will be saved. If no output is
             given, don't expect it will save anything. If by any change tries
             to save something, it will probably throw an error.
            `model_state`: Path to load stored parameters.
        """
        # Bring the model to GPU
        self.device = device
        self.out_path = output
        self.to(self.device)

        # Load or initialize weights
        if model_state != "":
            print("Loading previous model")
            self.load_state_dict(torch.load(model_state, map_location=self.device))
        else:
            def weight_init(m):
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d):
                    he_normal(m.weight)
                    torch.nn.init.zeros_(m.bias)
            self.apply(weight_init)

    def fit(self, tr_loader, val_loader, epochs, val_interval,
            loss, val_metrics, opt):
        """Trains the NN.

           Args:
            `tr_loader`: DataLoader with the training set.
            `val_loader`: DataLoader with the validaiton set.
            `epochs`: Number of epochs to train the model. If 0, no train.
            `val_interval`: After how many epochs to perform validation.
            `loss`: Name of the loss function.
            `val_metrics`: Which metrics to measure at validation time.
            `opt`: Optimizer.
        """
        t0 = time.time()
        e = 1
        # Expected classes of our dataset
        measure_classes = {0: "background", 1: "contra", 2: "R_hemisphere"}
        # Which classes will be reported during validation
        measure_classes_mean = np.array([1, 2])

        while e <= epochs:
            self.train()

            tr_loss = 0
            for (tr_i), (X, Y, info, W) in enumerate(tr_loader):
                X = [x.to(self.device) for x in X]
                Y = [y.to(self.device) for y in Y]
                W = [w.to(self.device) for w in W]

                output = self(X)
                pred = output[0]

                tr_loss_tmp = loss(output, Y, W)
                tr_loss += tr_loss_tmp

                # Optimization
                opt.zero_grad()
                tr_loss_tmp.backward()
                opt.step()

            tr_loss /= len(tr_loader)

            if len(val_loader) != 0 and e % val_interval == 0:
                log("Validation", self.out_path)
                self.eval()

                val_loss = 0
                # val_scores stores all needed metrics for assessing validation
                val_scores = np.zeros((len(val_metrics), len(val_loader), len(measure_classes)))
                Measure = Metric(val_metrics, onehot=softmax2onehot,
                        classes=measure_classes, multiprocess=False)

                with torch.no_grad():
                    for (val_i), (X, Y, info, W) in enumerate(val_loader):
                        X = [x.to(self.device) for x in X]
                        Y = [y.to(self.device) for y in Y]
                        W = [w.to(self.device) for w in W]

                        output = self(X)
                        val_loss_tmp = loss(output, Y, W)
                        val_loss += val_loss_tmp

                        y_true_cpu = Y[0].cpu().numpy()
                        y_pred_cpu = output[0].cpu().numpy()

                        # Record all needed metrics
                        # If batch_size > 1, Measure.all() returns an avg.
                        tmp_res = Measure.all(y_pred_cpu, y_true_cpu)
                        for i, m in enumerate(val_metrics):
                            val_scores[i, val_i] = tmp_res[m]

                # Validation loss
                val_loss /= len(val_loader)
                val_str = " Val Loss: {}".format(val_loss)

                # val_metrics shape: num_metrics x num_batches x num_classes
                for i, m in enumerate(val_metrics):
                    # tmp shape: num_classes (averaged over num_batches when val != -1)
                    tmp = np.array(Measure._getMean(val_scores[i]))

                    # Mean validation value in metric m (all interesting classes)
                    tmp_val = tmp[measure_classes_mean]
                    # Note: if tmp_val is NaN, it means that the classes I am
                    # interested in (check lib/data/whatever, measure_classes_mean)
                    # were not found in the validation set.
                    tmp_val = np.mean(tmp_val[tmp_val != -1])
                    val_str += ". Val " + m + ": " + str(tmp_val)

            else:
                val_str = ""

            eta = " ETA: " + datetime.fromtimestamp(time.time() + (epochs-e)*(time.time()-t0)/e).strftime("%Y-%m-%d %H:%M:%S")
            log("Epoch: {}. Loss: {}.".format(e, tr_loss) + val_str + eta, self.out_path)

            # Save model after every epoch
            torch.save(self.state_dict(), self.out_path + "model/MedicDeepLabv3Plus-model-" + str(e))
            if e > 1 and os.path.exists(self.out_path + "model/MedicDeepLabv3Plus-model-"+str(e-1)):
                os.remove(self.out_path + "model/MedicDeepLabv3Plus-model-" + str(e-1))

            e += 1

    def evaluate(self, test_loader, metrics, remove_islands, save_output=True):
        """Tests/Evaluates the NN.

           Args:
            `test_loader`: DataLoader containing the test set. Batch_size = 1.
            `metrics`: Metrics to measure.
            `save_output`: (bool) whether to save the output segmentations.
            `remove_islands`: (bool) whether to apply post-processing.
        """

        # Expected classes of our dataset
        measure_classes = {0: "background", 1: "contra", 2: "R_hemisphere"}

        results = {}
        self.eval()
        Measure = Metric(metrics, onehot=sigmoid2onehot,
                classes=measure_classes,
                multiprocess=True)

        # Pool to store pieces of output that will be put together
        # before evaluating the whole image.
        # This is useful when the entire image doesn't fit into mem.
        with torch.no_grad():
            for (test_i), (X, Y, info, W) in enumerate(test_loader):
                print("{}/{}".format(test_i+1, len(test_loader)))
                X = [x.to(self.device) for x in X]
                Y = [y.to(self.device) for y in Y]
                W = [w.to(self.device) for w in W]
                id_ = info["id"][0]

                output = self(X)

                y_pred_cpu = output[0].cpu().numpy()
                y_true_cpu = Y[0].cpu().numpy()

                if remove_islands:
                    y_pred_cpu = removeSmallIslands(y_pred_cpu, thr=20)

                # Predictions (and GT) separate the two hemispheres
                # combineLabels will combine these such that it creates
                # brainmask and contra-hemisphere ROIs instead of
                # two different hemisphere ROIs.
                y_pred_cpu = combineLabels(y_pred_cpu)
                # If GT was provided it measures the performance
                if len(y_true_cpu.shape) > 1:
                    y_true_cpu = combineLabels(y_true_cpu)

                    results[id_] = Measure.all(y_pred_cpu, y_true_cpu)

                test_loader.dataset.save(y_pred_cpu[0], info,
                        self.out_path + id_)

        # Gather results (multiprocessing)
        for k in results:
            results[k] = results[k].get()

        if len(results) > 0:
            with open(self.out_path + "stats.json", "w") as f:
                f.write(json.dumps(results))

        # If we are using multiprocessing we need to close the pool
        Measure.close()

