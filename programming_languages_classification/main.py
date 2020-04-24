# /usr/bin/env python3

import os
import sys
from dataset_loader import utils

##

PLDataset = utils.PLCExamplesDataset()

##

if __name__ == "__main__":
    # copy the dataset
    PLDataset.load()

