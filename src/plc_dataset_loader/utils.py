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
        counter = 0

        for directory in os.listdir(absolutePath):
            directory = str(directory)
            if directory.lower() in [x.lower() for x in ConfigurationManager.getLanguages()]:
                print(directory, end='\n')
                counter = counter + 1

        print("\n \n counter -> " + str(counter))
