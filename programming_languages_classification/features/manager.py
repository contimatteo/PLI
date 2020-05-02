# /usr/bin/env python3

import os
import json
from .parser import Parser
from .dictionary import DictionaryGenerator
from utils import ConfigurationManager, FileManager

FEATURE_FREQUENCY_THRESHOLD: int = ConfigurationManager.configuration['FEATURE_FREQUENCY_THRESHOLD']


class FeaturesManager:

    def __init__(self):
        self.DATASET_URI: str = "NOT_FOUND"
        self.N_NETWORK_PREFIX: str = "MISSING"

    def initialize(self, nNetworkPrefix: str, datasetUri: str):
        self.N_NETWORK_PREFIX = str(nNetworkPrefix)
        self.DATASET_URI = str(datasetUri)

    ##

    def parse(self):
        for languageFolder in FileManager.getLanguagesFolders(self.DATASET_URI):
            for exampleFolder in FileManager.getExamplesFolders(languageFolder.path):
                originalFileUri = FileManager.getOriginalFileUrl(exampleFolder.path)
                parsedFileUri = FileManager.getParsedFileUrl(exampleFolder.path)
                # create parsed file content
                if not os.path.exists(parsedFileUri):
                    parser = Parser()
                    parser.initialize(originalFileUri, parsedFileUri)
                    parser.parse()

    def generateFeatures(self, calculateWordsFrequency=True):
        # foreach language
        for languageFolder in FileManager.getLanguagesFolders(self.DATASET_URI):
            language = str(languageFolder.name).lower()
            wordInExamplesCounter: dict = {}
            wordFrequencies: dict = {}

            # check if 'features' file already exists
            featuresFileUri = FileManager.getFeaturesMapFileUrl(languageFolder.path)
            if os.path.exists(featuresFileUri):
                continue

            # generate the dictionary at 'example' level
            for exampleFolder in FileManager.getExamplesFolders(languageFolder.path):
                # parsed file uri
                parsedFileUri = FileManager.getParsedFileUrl(exampleFolder.path)
                # dictionary file uri
                dictionaryFileUri = FileManager.getDictionaryFileUrl(exampleFolder.path)
                # generate dictionary file content
                dictionaryGenerator = DictionaryGenerator()
                dictionaryGenerator.initialize(parsedFileUri, dictionaryFileUri)
                dictionaryGenerator.generate()

            if not calculateWordsFrequency:
                continue

            # repeating: count '+1' if a word is contained in this example
            for exampleFolder in FileManager.getExamplesFolders(languageFolder.path):
                # read dictionary file
                dictionaryFileUri = FileManager.getDictionaryFileUrl(exampleFolder.path)
                dictionaryContent = json.loads(FileManager.readFile(dictionaryFileUri))
                # count all words in this exmaple
                for word in dictionaryContent['words']:
                    if word not in wordInExamplesCounter:
                        wordInExamplesCounter[word] = 1
                    else:
                        wordInExamplesCounter[word] += 1

            # calculate and filter word frequency (foreach language)
            for word in wordInExamplesCounter:
                frequency = wordInExamplesCounter[word] / ConfigurationManager.getLanguagesExamplesCounter()[language]
                if frequency > FEATURE_FREQUENCY_THRESHOLD:
                    wordFrequencies[word] = frequency

            # saving the dictionary at 'language' level
            FileManager.writeFile(featuresFileUri, json.dumps({'words_frequencies': wordFrequencies}))
