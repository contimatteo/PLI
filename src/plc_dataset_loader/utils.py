import os
import json
from configurations.manager import Manager as ConfigurationManager


class PLCExamplesDataset:
    ORIGINAL_DATASET_URI: str = "datasets/rosetta-code/Lang"
    DESTINATION_DATASET_URI = "src/data/datasets"
    DATASET_TESTING_FOLDER_NAME = 'testing'
    DATASET_TRAINING_FOLDER_NAME = 'training'

    #

    def load(self):
        absolutePath = os.getcwd() + '/' + self.ORIGINAL_DATASET_URI

        for directory in os.listdir(absolutePath):
            if directory in ConfigurationManager.getLanguages():
                print(directory, end='\n')
