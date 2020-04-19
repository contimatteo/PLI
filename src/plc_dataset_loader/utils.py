import os
import json
from plc_configurations.manager import Manager as ConfigurationManager


EXAMPLES_NUMBER_THRESHOLD = 400


class PLCExamplesDataset:

    ORIGINAL_DATASET_URI: str = "datasets/rosetta-code/Lang"
    DESTINATION_DATASET_URI = "src/data/datasets"
    DATASET_TESTING_FOLDER_NAME = 'testing'
    DATASET_TRAINING_FOLDER_NAME = 'training'

    def load(self):
        absolutePath = os.path.join(os.getcwd(), self.ORIGINAL_DATASET_URI)

        for languageDir in os.listdir(absolutePath):
            languageDir = str(languageDir)
            numberOfExamples = 0

            if languageDir.lower() in [x.lower() for x in ConfigurationManager.getLanguages()]:
                languageDirectoryAbsoluteUri = os.path.join(absolutePath, languageDir)

                for exampleDir in [f.path for f in os.scandir(languageDirectoryAbsoluteUri) if f.is_dir()]:
                    # count each version (file) provided for this example
                    # numberOfExamples += len(os.listdir(os.path.join(languageDirectoryAbsoluteUri, exampleDir)))
                    # skipping multiple example versions
                    numberOfExamples += 1

                if numberOfExamples < EXAMPLES_NUMBER_THRESHOLD:
                    print("[INFO] "+languageDir+" ("+str(numberOfExamples)+" examples)")








