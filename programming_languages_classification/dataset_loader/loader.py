# /usr/bin/env python3

import os
import sys
import shutil
from configurations import ConfigurationManager


ROOT_DIR: str = os.path.abspath(os.path.dirname(sys.argv[0]))
EXAMPLES_NUMBER_THRESHOLD = 400
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

        # # delete training dataset if exists
        # if os.path.isdir(self.TRAINING_ABS_URI):
        #     shutil.rmtree(self.TRAINING_ABS_URI)
        # # delete testing dataset if exists
        # if os.path.isdir(self.TESTING_ABS_URI):
        #     shutil.rmtree(self.TESTING_ABS_URI)

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

                examplesForLanguageCounter = 0
                # list all examples in {languageFolder.name} folder
                for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:

                    # list all examples versions in {exampleFolder.name} folder
                    for exampleVersionFile in [f for f in os.scandir(exampleFolder.path) if f.is_file()]:
                        examplesForLanguageCounter += 1

                        # TODO: create random function ...
                        # move file to right dataset
                        if examplesForLanguageCounter > EXAMPLES_NUMBER_THRESHOLD:
                            DATASET_TYPE = self.TESTING_ABS_URI
                        else:
                            DATASET_TYPE = self.TRAINING_ABS_URI

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
                        # print("Copying file to --> " + destinationFileUri.replace(ROOT_DIR, ''))
                        shutil.copyfile(exampleVersionFile.path, destinationFileUri)
