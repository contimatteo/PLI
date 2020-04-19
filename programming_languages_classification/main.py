# /usr/bin/env python3

from plc_dataset_loader import utils

##

PLDataset = utils.PLCExamplesDataset()

##

if __name__ == "__main__":
    print("\n")
    PLDataset.load()
    print("\n")
