# python eval.py --input /home/miguelv/data/in/validation/ --output /home/miguelv/data/out/delete/test/38/ --model /home/miguelv/data/in/models/model-1

import os, time, torch, json
import numpy as np
import nibabel as nib
from lib.utils import *
from lib.losses import Loss
from torch.utils.data import DataLoader
from datetime import datetime
from lib.models.MedicDeepLabv3Plus import MedicDeepLabv3Plus
from lib.data.DataWrapper import DataWrapper
from lib.metric import Metric

def get_arguments():
    """Gets (and parses) the arguments from the command line.

       Args:
        `args`: If None, it takes the arguments from the command line.
                Else, it will parse `args` (used for testing with sacred)
    """
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
           return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
           return False
        else:
           raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--input", type=str, required=True,
            help="Directory with the data for generating new masks")
    parser.add_argument("--model", dest="model", default=-1, nargs="+",
            required=True,
            help="Files with the saved parameters (output model-X of train.py)")

    # Evaluation
    parser.add_argument("--metrics", type=str, default="dice",
            help="List of which metrics to measure at evaluation time")
    parser.add_argument("--remove_islands", type=str2bool, nargs="?",
            default=True,
            help="Whether to apply the post-processing operation.")
    parser.add_argument("--filters", type=int, default=32,
            help="Number of filters (fewer filters -> lower GPU requirements)")

    # Other
    parser.add_argument("--output", type=str, required=True,
            help="Output directory (if it doesn't exist, it will create it)")
    parser.add_argument("--gpu", type=int, default=0, dest="device",
            help="GPU Device. Write -1 if no GPU is available")
 
    parsed = parser.parse_args()

    # --input
    if not os.path.isdir(parsed.input):
        raise Exception("The input folder `" + parsed.input + "` does not exist")

    # --model
    if parsed.model == -1:
        raise Exception("Provide the `model` file (--model FOLDER/model-*)")
    else:
        if type(parsed.model) == list:
            for l in parsed.model:
                if not os.path.isfile(l):
                    raise Exception("The provided file `"+l+"` does not exist or it's not a file")
        else:
            if not os.path.isfile(parsed.model):
                raise Exception("The provided file `"+parsed.model+"` does not exist or it's not a file")

    # --gpu
    if parsed.device >= torch.cuda.device_count():
        if torch.cuda.device_count() == 0:
            print("> No available GPUs. Add --gpu -1 to not use GPU. NOTE: This may take FOREVER to run.")
        else:
            print("> Available GPUs:")
            for i in range(torch.cuda.device_count()):
                print("    > GPU #"+str(i)+" ("+torch.cuda.get_device_name(i)+")")
        raise Exception("The GPU #"+str(parsed.device)+" does not exist. Check available GPUs.")

    # --output
    if os.path.exists(parsed.output):
        if os.path.isfile(parsed.output):
            raise Exception("The provided path for the --output `" + parsed.output + "` corresponds to an existing file. Provide a non-existing path or a folder.")
        elif os.path.isdir(parsed.output):
            files = [int(f) for f in os.listdir(parsed.output) if f.isdigit()]
            parsed.output = os.path.join(parsed.output, str(len(files)+1), "")
            os.makedirs(parsed.output)
        else:
            raise Exception("The provided path for the --output `" + parsed.output + "` is invalid. Provide a non-existing path or a folder.")
    else:
        parsed.output = os.path.join(parsed.output, "1", "")
        os.makedirs(parsed.output)

    if parsed.device > -1:
        parsed.device = "cuda:"+str(parsed.device)
    else:
        parsed.device = "cpu"

    # Metrics to be evaluated during evaluation
    allowed_metrics = ["dice", "HD", "compactness"]
    parsed.metrics = parsed.metrics.split(",")
    for m in parsed.metrics:
        if not m in allowed_metrics:
            raise Exception("Wrong --metrics: "+str(m)+". Only allowed: "+str(allowed_metrics))

    return parsed


def main(args):
    # "record" won't do anything. I keep this variable to record
    # the transformations.

    # Creates the output folder
    #os.makedirs(args.output)

    for i, model_state in enumerate(args.model):
        log("Generating masks with model " + str(i+1), args.output)

        os.makedirs(args.output + str(i+1))

        model = MedicDeepLabv3Plus(modalities=1, n_classes=3, first_filters=args.filters)
        model.initialize(device=args.device, output=args.output + str(i+1) + "/",
                model_state=model_state)

        # Dataloaders
        test_data = DataWrapper(args.input, "test")

        if len(test_data) > 0:
            test_loader = DataLoader(test_data, batch_size=1,
                    shuffle=False, pin_memory=args.device != "cpu",
                    num_workers=1)

            model.evaluate(test_loader=test_loader, metrics=args.metrics,
                    remove_islands=args.remove_islands)

    # Majority voting
    if len(args.model) > 1:
        print("Majority voting:")
        os.makedirs(args.output + "majorityVoting/")

        measure_classes = {0: "background", 1: "contra", 2: "R_hemisphere"}
        Measure = Metric(args.metrics, onehot=sigmoid2onehot,
                classes=measure_classes,
                multiprocess=True)
        results = {}

        for (te_i), (_, Y, info, _) in enumerate(test_loader):
            id_ = info["id"][0]
            Y = Y[0].numpy()
            print("{}/{}".format(te_i+1, len(test_loader)))

            brains = []
            for i in range(len(args.model)):
                brainmask = nib.load(args.output + str(i+1) + "/" + id_ + "_brainmask.nii.gz").get_data().transpose(2, 0, 1)
                contra = nib.load(args.output + str(i+1) + "/" + id_ + "_contra.nii.gz").get_data().transpose(2, 0, 1)
                background = 1 - brainmask

                brain = np.stack([background, brainmask, contra])
                brains.append(brain)
            brains = np.stack(brains)
            result = 1.0*(np.sum(brains, axis=0) > (len(args.model)/2))

            # Saving majority voting
            test_data.save(result, info, args.output + "majorityVoting/" + id_) 
         
            if len(Y.shape) > 1:
                Y = combineLabels(Y)
                result = np.expand_dims(result, axis=0)

                results[id_] = Measure.all(result, Y)

        for k in results:
            results[k] = results[k].get()

        if len(results) > 0:
            with open(args.output + "majorityVoting/stats.json", "w") as f:
                f.write(json.dumps(results))

        Measure.close()

    log("End", args.output)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
