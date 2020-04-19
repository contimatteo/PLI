import os
import json
from shutil import copyfile
from plc_configurations.manager import Manager as ConfigurationManager


EXAMPLES_NUMBER_THRESHOLD = 400


class PLCExamplesDataset:

    SOURCE_FOLDER: str = "datasets/rosetta-code/Lang"
    DESTINATION_FOLDER = "src/data/datasets"
    TRAINING_FOLDER = 'testing'
    TESTING_FOLDER = 'training'

    SOURCE_ABS_URI: str = None
    TRAINING_ABS_URI = None
    TESTING_ABS_URI = None

    ##

    def __init__(self):
        self.SOURCE_ABS_URI = os.path.join(os.getcwd(), self.SOURCE_FOLDER)
        self.TRAINING_ABS_URI = os.path.join(os.getcwd(), self.DESTINATION_FOLDER, self.TRAINING_FOLDER)
        self.TESTING_ABS_URI = os.path.join(os.getcwd(), self.DESTINATION_FOLDER, self.TESTING_FOLDER)

    def load(self):
        # foreach directory in '/Lang' folder ...
        for languageFolder in [f for f in os.scandir(self.SOURCE_ABS_URI) if f.is_dir()]:
            # parse only selected languages
            if str(languageFolder.name).lower() in [x.lower() for x in ConfigurationManager.getLanguages()]:
                # prepraing empty {languageFolder.name} folder
                destinationFolder = str(languageFolder.name).lower()
                if not(os.path.isdir(os.path.join(self.TESTING_ABS_URI, destinationFolder))):
                    os.mkdir(os.path.join(self.TESTING_ABS_URI, destinationFolder))

                # list all examples in {languageFolder.name} folder
                for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:
                    # list all examples versions in {exampleFolder.name} folder
                    for exampleVersionFile in [f for f in os.scandir(exampleFolder.path) if f.is_file()]:
                        destinationFileName = str(exampleVersionFile.name).lower() + '.txt'
                        destinationFileUri = os.path.join(self.TESTING_ABS_URI, destinationFolder, destinationFileName)
                        # create an empty file
                        file = open(destinationFileUri, "a+")
                        file.close()
                        # copy the original source file content
                        print(destinationFileUri)
                        copyfile(exampleVersionFile.path, destinationFileUri)

code
