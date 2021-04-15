# Example usage:
# python train.py --device cuda --epochs 10 --input /home/miguelv/data/in/train/ --output /home/miguelv/data/out/delete/test/25/

import os, time, torch, json
import numpy as np
import nibabel as nib
from lib.utils import *
from lib.losses import Loss
from torch.utils.data import DataLoader
from datetime import datetime
from lib.models.MedicDeepLabv3Plus import MedicDeepLabv3Plus
from lib.data.DataWrapper import DataWrapper

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
            help="Directory with the data for optimizing MedicDeepLabv3+")

    # Training
    parser.add_argument("--epochs", type=int, default=300,
            help="Epochs. If 0: only evaluate")
    parser.add_argument("--batch_size", type=int, default=1,
            help="Batch size")
    parser.add_argument("--lr", type=float, default="1e-4",
            help="Learning rate")
    parser.add_argument("--wd", type=float, default="0",
            help="Weight decay")
    parser.add_argument("--filters", type=int, default=32,
            help="Number of filters (fewer filters -> lower GPU requirements)")

    # Validation
    parser.add_argument("--validation", type=str, default="",
            help="Directory with the data for validation")
    parser.add_argument("--val_interval", type=int, default=1,
            help="After how many epochs data is validated")
    parser.add_argument("--val_metrics", type=str, default="dice",
            help="List of metrics to measure during validation")

    # Other
    parser.add_argument("--output", type=str, required=True,
            help="Output directory (if it doesn't exist, it will create it)")
    parser.add_argument("--gpu", type=int, default=0, dest="device",
            help="GPU Device. Write -1 if no GPU is available")
    parser.add_argument("--model_state", type=str, default="",
            help="File that contains the saved parameters of the model")
 
    parsed = parser.parse_args()

    # --input
    if not os.path.isdir(parsed.input):
        raise Exception("The input folder `" + parsed.input + "` does not exist")

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


    # --validation
    if parsed.validation != "" and not os.path.isdir(parsed.validation):
        raise Exception("The validaiton folder `" + parsed.validation + "` does not exist")
    if parsed.validation == "":
        print("> Note: No validation data was provided, so validation won't be done during MedicDeepLabv3+ optimization")

    # --gpu
    if parsed.device >= torch.cuda.device_count():
        if torch.cuda.device_count() == 0:
            print("> No available GPUs. Add --gpu -1 to not use GPU. NOTE: This may take FOREVER to run.")
        else:
            print("> Available GPUs:")
            for i in range(torch.cuda.device_count()):
                print("    > GPU #"+str(i)+" ("+torch.cuda.get_device_name(i)+")")
        raise Exception("The GPU #"+str(parsed.device)+" does not exist. Check available GPUs.")

    if parsed.device > -1:
        parsed.device = "cuda:"+str(parsed.device)
    else:
        parsed.device = "cpu"

    # Metrics to be evaluated during evaluation
    allowed_metrics = ["dice", "HD", "compactness"]

    # Metrics to be evaluated during validation
    parsed.val_metrics = parsed.val_metrics.split(",")
    for m in parsed.val_metrics:
        if not m in allowed_metrics:
            raise Exception("Wrong --val_metrics: "+str(m)+". Only allowed: "+str(allowed_metrics))

    return parsed


def main(args):

    log("Start training MedicDeepLabv3+", args.output)

    # Creates the folder where the models will be saved
    os.makedirs(args.output + "model")

    # Parameters required to initialize the model
    model = MedicDeepLabv3Plus(modalities=1, n_classes=3, first_filters=args.filters)
    model.initialize(device=args.device, output=args.output,
            model_state=args.model_state)

    # Dataloaders
    tr_data = DataWrapper(args.input, "train")
    val_data = DataWrapper(args.validation, "val")

    if len(tr_data) > 0 and args.epochs > 0:
        # DataLoaders
        tr_loader = DataLoader(tr_data, batch_size=args.batch_size,
                shuffle=True, pin_memory=False, num_workers=6)

        if len(val_data) > 0:
            val_loader = DataLoader(val_data, batch_size=args.batch_size,
                    shuffle=False, pin_memory=False, num_workers=6)
        else:
            val_loader = [] # So that len(val_loader) = 0

        # Loss function
        loss = Loss("CrossEntropyDiceLoss_multiple") # Deep supervision

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                weight_decay=args.wd)

        # Train the model
        model.fit(tr_loader=tr_loader, val_loader=val_loader,
                epochs=args.epochs, val_interval=args.val_interval,
                loss=loss, val_metrics=args.val_metrics, opt=optimizer)

    log("End", args.output)


if __name__ == "__main__":
    # Get command-line arguments
    args = get_arguments()

    # Train MedicDeepLabv3+
    main(args)
