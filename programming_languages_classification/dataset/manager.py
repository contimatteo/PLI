# /usr/bin/env python3

import os
import json
import shutil
import random
from utils import ConfigurationManager, FileManager
from .instance import DatasetInstance
from features import Parser


ESCAPED_TOKENS = ConfigurationManager.escaped_tokens
TRAINING_EXAMPLES_NUMBER: int = ConfigurationManager.configuration['TRAINING_EXAMPLES_NUMBER']

# backup vars
tmpTokenOccurrencies = None
tmpTokensWithLowRelevance = None


class DatasetManager:
    Dataset = None
    algorithm: str = None

    def __init__(self):
        self.Dataset = DatasetInstance()

    def initialize(self, algorithm: str):
        self.Dataset.initialize()
        self.algorithm = algorithm

        return self

    def load(self):
        datasetAlreadyExists = self.__create_folders()

        # clone file sources if dataset doesn't already exists
        if not datasetAlreadyExists:
            self.__cloneFilesSources()

        # if not os.path.exists(FileManager.getDatasetCopyFileUrl()):
        # load dataset in memory
        self.__loadInMemory()
            # generate 'filtered' version
        #     self.__filterSources()
        #
        #     # save dataset copy
        #     datasetCopy: dict = {'training': self.Dataset.training,  'testing': self.Dataset.testing}
        #     FileManager.writeFile(FileManager.getDatasetCopyFileUrl(), json.dumps(datasetCopy))
        # else:
        #     datasetCopy = json.loads(FileManager.readFile(FileManager.getDatasetCopyFileUrl()))
        #     self.Dataset.training = datasetCopy['training']
        #     self.Dataset.testing = datasetCopy['testing']

        return self

    def __create_folders(self):
        datasetAlreadyExists = True

        # initialize dataset folders

        if not os.path.isdir(FileManager.datasets['training']['url']):
            datasetAlreadyExists = False
            os.mkdir(FileManager.datasets['training']['url'])

        if not os.path.isdir(FileManager.datasets['testing']['url']):
            datasetAlreadyExists = False
            os.mkdir(FileManager.datasets['testing']['url'])

        # initialize output folders

        if not os.path.isdir(FileManager.getFeaturesFolderUrl()):
            os.mkdir(FileManager.getFeaturesFolderUrl())

        if not os.path.isdir(FileManager.getModelsFolderUrl()):
            os.mkdir(FileManager.getModelsFolderUrl())

        if not os.path.isdir(FileManager.getVocabularyFolderUrl()):
            os.mkdir(FileManager.getVocabularyFolderUrl())

        if not os.path.isdir(FileManager.getReportsFolderUrl()):
            os.mkdir(FileManager.getReportsFolderUrl())

        return datasetAlreadyExists

    def __cloneFilesSources(self):
        SOURCE_URL = FileManager.datasets['source']['url']
        TRAINING_URL = FileManager.datasets['training']['url']
        TESTING_URL = FileManager.datasets['testing']['url']

        # foreach directory in '/Lang' folder ...
        languagesExamplesCounter = {}
        for languageFolder in [f for f in os.scandir(SOURCE_URL) if f.is_dir()]:
            language = str(languageFolder.name).lower()
            languagesExamplesCounter[language] = 0
            # parse only selected languages
            if language in ConfigurationManager.getLanguages():
                # preparing empty {languageFolder.name} for each dataset
                if not (os.path.isdir(os.path.join(TRAINING_URL, language))):
                    os.mkdir(os.path.join(TRAINING_URL, language))
                if not (os.path.isdir(os.path.join(TESTING_URL, language))):
                    os.mkdir(os.path.join(TESTING_URL, language))

                # count example foreach language
                for exampleFolder in FileManager.getExamplesFolders(languageFolder.path):
                    for _ in FileManager.getExampleFiles(exampleFolder.path):
                        languagesExamplesCounter[language] += 1

                # print languages with examples counter less than {TRAINING_EXAMPLES_NUMBER}
                if languagesExamplesCounter[language] < TRAINING_EXAMPLES_NUMBER:
                    print(' >  [dataset] the total number of examples for the '
                          + language + ' is less than ' + str(TRAINING_EXAMPLES_NUMBER))
                    continue

                # for this language, the total examples number could be less than {TRAINING_EXAMPLES_NUMBER}
                indexesOfTrainingExamples = random.sample(
                    range(1, languagesExamplesCounter[language]),
                    TRAINING_EXAMPLES_NUMBER
                )

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
                        # copy the ORIGINAL source file content
                        originalFileUri = FileManager.getOriginalFileUrl(exampleFolderUri)
                        FileManager.createFile(originalFileUri)
                        shutil.copyfile(exampleVersionFile.path, originalFileUri)
                        # create the  'PARSED' version of the orginal file
                        parsedFileUri = FileManager.getParsedFileUrl(exampleFolderUri)
                        FileManager.createFile(parsedFileUri)
                        parser = Parser()
                        parser.initialize(originalFileUri, parsedFileUri)
                        parser.parse()

        return self

    def __loadInMemory(self):
        TRAINING_URL = FileManager.datasets['training']['url']
        TESTING_URL = FileManager.datasets['testing']['url']

        # training
        for languageFolder in FileManager.getLanguagesFolders(TRAINING_URL):
            language = str(languageFolder.name).lower()
            self.Dataset.addLanguage('training', language)
            # example
            for exampleFolder in FileManager.getExamplesFolders(languageFolder.path):
                exampleDict: dict = {}
                # original file
                originalFileUri = FileManager.getOriginalFileUrl(exampleFolder.path)
                originalFileContent = FileManager.readFile(originalFileUri)
                exampleDict['original'] = originalFileContent
                # parsed file
                parsedFileUri = FileManager.getParsedFileUrl(exampleFolder.path)
                parsedFileContent = FileManager.readFile(parsedFileUri)
                exampleDict['parsed'] = parsedFileContent
                # save
                self.Dataset.addExample('training', language, exampleDict)

        # testing
        for languageFolder in FileManager.getLanguagesFolders(TESTING_URL):
            language = str(languageFolder.name).lower()
            self.Dataset.addLanguage('testing', language)
            # example
            for exampleFolder in FileManager.getExamplesFolders(languageFolder.path):
                exampleDict: dict = {}
                # original file
                originalFileUri = FileManager.getOriginalFileUrl(exampleFolder.path)
                originalFileContent = FileManager.readFile(originalFileUri)
                exampleDict['original'] = originalFileContent
                # parsed file
                parsedFileUri = FileManager.getParsedFileUrl(exampleFolder.path)
                parsedFileContent = FileManager.readFile(parsedFileUri)
                exampleDict['parsed'] = parsedFileContent
                # save
                self.Dataset.addExample('testing', language, exampleDict)

        return self

    def __filterSources(self):
        global tmpTokenOccurrencies
        global tmpTokensWithLowRelevance

        tokenOccurrencies: dict = {}
        tokensWithLowRelevance = {}
        relevanceOccurencyThreshold = 2

        # count tokens occurrency
        if tmpTokenOccurrencies is None:
            sources = self.Dataset.getSources('training')
            for language in sources:
                if language not in tokenOccurrencies:
                    tokenOccurrencies[language] = {}
                for exampleDict in sources[language]:
                    source = str(exampleDict['parsed'])
                    source = source.replace(ESCAPED_TOKENS['ALPHA'], '')
                    source = source.replace(ESCAPED_TOKENS['NUMBER'], '')
                    source = source.replace('\n', ' ')
                    # set(...) avoid counting occurrency inside this single example
                    for token in set(source.split(' ')):
                        if token not in tokenOccurrencies[language]:
                            tokenOccurrencies[language][token] = 1
                        else:
                            tokenOccurrencies[language][token] += 1

            tmpTokenOccurrencies = tokenOccurrencies
        else:
            tokenOccurrencies = tmpTokenOccurrencies

        # extract tokens with occurrency lower than a threshold
        if tmpTokensWithLowRelevance is None:
            for language in sources:
                if language not in tokensWithLowRelevance:
                    tokensWithLowRelevance[language] = []
                for token in tokenOccurrencies[language]:
                    occurency = tokenOccurrencies[language][token]
                    if occurency < relevanceOccurencyThreshold:
                        tokensWithLowRelevance[language].append(token)

            tmpTokensWithLowRelevance = tokensWithLowRelevance
        else:
            tokensWithLowRelevance = tmpTokensWithLowRelevance

        # replace token with low relevance (selected in the previou step)
        sources = self.Dataset.getSources('training')
        for language in sources:
            for exampleDict in sources[language]:
                source = str(exampleDict['parsed'])
                source = source.replace('\n', ' ')
                exampleDict['filtered'] = ''
                for token in source.split(' '):
                    exampleDict['filtered'] += ' '
                    if token not in tokensWithLowRelevance[language]:
                        exampleDict['filtered'] += str(token).replace(' ', '')
                    else:
                        exampleDict['filtered'] += ESCAPED_TOKENS['NOT_RELEVANT']

        # clone testing 'parsed' sources as 'filtered' (no changes applied)
        sources = self.Dataset.getSources('testing')
        for language in sources:
            for exampleDict in sources[language]:
                exampleDict['filtered'] = str(exampleDict['parsed'])

        return self
