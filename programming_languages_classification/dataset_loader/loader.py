# /usr/bin/env python3

import os
import sys
import shutil
import random
from configurations import ConfigurationManager


ROOT_DIR: str = os.path.abspath(os.path.dirname(sys.argv[0]))
TRAINING_EXAMPLES_NUMBER: int = 400
FILE_NAMES: dict = ConfigurationManager.getFileNames()


class DatasetLoader:
    SOURCE_FOLDER: str = "../datasets/rosetta-code/Lang"

    DESTINATION_FOLDER: str = "data"
    TRAINING_FOLDER: str = 'training'
    TESTING_FOLDER: str = 'testing'

    SOURCE_ABS_URI: str = ''
    TRAINING_ABS_URI: str = ''
    TESTING_ABS_URI: str = ''

    ##

    def __init__(self):
        self.SOURCE_ABS_URI = os.path.join(ROOT_DIR, self.SOURCE_FOLDER)
        self.TRAINING_ABS_URI = os.path.join(ROOT_DIR, self.DESTINATION_FOLDER, self.TRAINING_FOLDER)
        self.TESTING_ABS_URI = os.path.join(ROOT_DIR, self.DESTINATION_FOLDER, self.TESTING_FOLDER)

    def create_folders(self):
        datasetAlreadyExists = True

        if not(os.path.isdir(os.path.join(ROOT_DIR, self.DESTINATION_FOLDER))):
            datasetAlreadyExists = False
            os.mkdir(os.path.join(ROOT_DIR, self.DESTINATION_FOLDER))

        # initialize training dataset folders
        if not(os.path.isdir(self.TRAINING_ABS_URI)):
            datasetAlreadyExists = False
            os.mkdir(self.TRAINING_ABS_URI)
        # initialize testing dataset folders
        if not(os.path.isdir(self.TESTING_ABS_URI)):
            datasetAlreadyExists = False
            os.mkdir(self.TESTING_ABS_URI)

        return datasetAlreadyExists

    def load(self):
        # create folders ...
        datasetAlreadyExists = self.create_folders()
        if datasetAlreadyExists:
            return

        # foreach directory in '/Lang' folder ...
        for languageFolder in [f for f in os.scandir(self.SOURCE_ABS_URI) if f.is_dir()]:
            # parse only selected languages
            if str(languageFolder.name).lower() in [x.lower() for x in ConfigurationManager.getLanguages()]:
                # preparing empty {languageFolder.name} folder into training dataset
                language = str(languageFolder.name).lower()
                if not(os.path.isdir(os.path.join(self.TRAINING_ABS_URI, language))):
                    os.mkdir(os.path.join(self.TRAINING_ABS_URI, language))
                # preparing empty {languageFolder.name} folder into testing dataset
                if not(os.path.isdir(os.path.join(self.TESTING_ABS_URI, language))):
                    os.mkdir(os.path.join(self.TESTING_ABS_URI, language))

                examplesNumberForThisLanguage = 0
                examplesPaths = [f for f in os.scandir(languageFolder.path) if f.is_dir()]

                # list all examples in {languageFolder.name} folder
                for exampleFolder in examplesPaths:
                    # list all examples versions in {exampleFolder.name} folder
                    for exampleVersionFile in [f for f in os.scandir(exampleFolder.path) if f.is_file()]:
                        examplesNumberForThisLanguage += 1

                if examplesNumberForThisLanguage < TRAINING_EXAMPLES_NUMBER:
                    print(' > [dataset] '+str(language)+' has examples number less than '+str(TRAINING_EXAMPLES_NUMBER))
                    continue

                # for this language, the total examples number could be less than {TRAINING_EXAMPLES_NUMBER}
                indexesOfTrainingExamples = random.sample(range(1, examplesNumberForThisLanguage), TRAINING_EXAMPLES_NUMBER)

                # list all examples in {languageFolder.name} folder
                exampleIndex = 0
                for exampleFolder in examplesPaths:
                    # list all examples versions in {exampleFolder.name} folder
                    for exampleVersionFile in [f for f in os.scandir(exampleFolder.path) if f.is_file()]:
                        exampleIndex += 1
                        # move file to right dataset
                        if exampleIndex in indexesOfTrainingExamples:
                            DATASET_TYPE = self.TRAINING_ABS_URI
                        else:
                            DATASET_TYPE = self.TESTING_ABS_URI

                        # prepare destination folder
                        example = str(exampleVersionFile.name).lower()
                        destinationFolderUri = os.path.join(DATASET_TYPE, language, example)
                        os.mkdir(destinationFolderUri)
                        # prepare destination path for this example (file.txt)
                        destinationFileUri = os.path.join(destinationFolderUri, FILE_NAMES['ORIGINAL'])
                        # create an empty file
                        file = open(destinationFileUri, "a+")
                        file.close()
                        # copy the original source file content
                        # Copying file to --> destinationFileUri.replace(ROOT_DIR, '')
                        shutil.copyfile(exampleVersionFile.path, destinationFileUri)
