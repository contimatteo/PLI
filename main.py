from plc_dataset_loader import utils

##

PLDataset = utils.PLCExamplesDataset()

print(__name__)

##

if __name__ == "__main__":
    PLDataset.load()
