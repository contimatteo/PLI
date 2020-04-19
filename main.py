#

from .loader import utils

PLDataset = utils.PLExamplesDataset()

if __name__ == 'main':
    PLDataset.load()
