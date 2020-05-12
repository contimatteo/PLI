# /usr/bin/env python3

import os
import shutil
import random
from utils import ConfigurationManager, FileManager


TRAINING_EXAMPLES_NUMBER: int = ConfigurationManager.configuration['TRAINING_EXAMPLES_NUMBER']


def create_folders():
    datasetAlreadyExists = True

    # initialize training dataset folders
    if not(os.path.isdir(FileManager.datasets['training']['url'])):
        datasetAlreadyExists = False
        os.mkdir(FileManager.datasets['training']['url'])
    # initialize testing dataset folders
    if not(os.path.isdir(FileManager.datasets['testing']['url'])):
        datasetAlreadyExists = False
        os.mkdir(FileManager.datasets['testing']['url'])

    return datasetAlreadyExists


class DatasetManager:

    def load(self):
        SOURCE_URL = FileManager.datasets['source']['url']
        TRAINING_URL = FileManager.datasets['training']['url']
        TESTING_URL = FileManager.datasets['testing']['url']

        # create folders ...
        datasetAlreadyExists = create_folders()

        # return if dataset already exists
        if datasetAlreadyExists:
            return self

        # foreach directory in '/Lang' folder ...
        languagesExamplesCounter = {}
        for languageFolder in [f for f in os.scandir(SOURCE_URL) if f.is_dir()]:
            language = str(languageFolder.name).lower()
            languagesExamplesCounter[language] = 0
            # parse only selected languages
            if language in ConfigurationManager.getLanguages():
                # preparing empty {languageFolder.name} for each dataset
                if not(os.path.isdir(os.path.join(TRAINING_URL, language))):
                    os.mkdir(os.path.join(TRAINING_URL, language))
                if not(os.path.isdir(os.path.join(TESTING_URL, language))):
                    os.mkdir(os.path.join(TESTING_URL, language))

                # count example foreach language
                for exampleFolder in FileManager.getExamplesFolders(languageFolder.path):
                    for _ in FileManager.getExampleFiles(exampleFolder.path):
                        languagesExamplesCounter[language] += 1

                # print languages with examples counter less than {TRAINING_EXAMPLES_NUMBER}
                if languagesExamplesCounter[language] < TRAINING_EXAMPLES_NUMBER:
                    print(' > [dataset] '+str(language)+' has examples number less than '+str(TRAINING_EXAMPLES_NUMBER))
                    continue

                # for this language, the total examples number could be less than {TRAINING_EXAMPLES_NUMBER}
                indexesOfTrainingExamples = random.sample(range(1, languagesExamplesCounter[language]), TRAINING_EXAMPLES_NUMBER)

                # list all examples in {languageFolder.name} folder
                exampleIndex = 0
                for exampleFolder in FileManager.getExamplesFolders(languageFolder.path):
                    # list all examples versions in {exampleFolder.name} folder
                    for exampleVersionFile in FileManager.getExampleFiles(exampleFolder.path):
                        exampleIndex += 1
                        # move file to right dataset
                        if exampleIndex in indexesOfTrainingExamples:
                            DATASET_TYPE = TRAINING_URL
                        else:
                            DATASET_TYPE = TESTING_URL

                        # prepare destination folder
                        example = str(exampleVersionFile.name).lower()
                        exampleFolderUri = os.path.join(DATASET_TYPE, language, example)
                        os.mkdir(exampleFolderUri)
                        # copy the original source file content
                        FileManager.createFile(FileManager.getOriginalFileUrl(exampleFolderUri))
                        shutil.copyfile(exampleVersionFile.path, FileManager.getOriginalFileUrl(exampleFolderUri))

        return self
