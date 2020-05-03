# /usr/bin/env python3

import os
import sys



ROOT_DIR: str = os.path.abspath(os.path.dirname(sys.argv[0]))
SOURCE_FOLDER: str = "../datasets/rosetta-code/Lang"
DESTINATION_FOLDER: str = "data"
TRAINING_FOLDER: str = 'training'
TESTING_FOLDER: str = 'testing'

FILE_NAMES: dict = {
    'ORIGINAL': 'original.txt',
    'PARSED': 'parsed.txt',
    'FEATURES': 'features.json',
    'DICTIONARY': 'dictionary.json',
}


class FileManagerClass:

    def __init__(self):
        self.datasets = {'source': {}, 'training': {}, 'testing': {}}
        self.initialize()

    def initialize(self):
        self.datasets['source']['url'] = os.path.join(ROOT_DIR, SOURCE_FOLDER)
        self.datasets['training']['url'] = os.path.join(ROOT_DIR, *[DESTINATION_FOLDER, TRAINING_FOLDER])
        self.datasets['testing']['url'] = os.path.join(ROOT_DIR, *[DESTINATION_FOLDER, TESTING_FOLDER])
        return self

    def getOriginalFileUrl(self, exampleFolderPath):
        return exampleFolderPath + '/' + FILE_NAMES['ORIGINAL']

    def getParsedFileUrl(self, exampleFolderPath):
        return exampleFolderPath + '/' + FILE_NAMES['PARSED']

    def getDictionaryFileUrl(self, exampleFolderPath):
        return exampleFolderPath + '/' + FILE_NAMES['DICTIONARY']

    def getFeaturesMapFileUrl(self, languageFolder):
        return languageFolder + '/' + FILE_NAMES['FEATURES']

    def getEncoderLabelsFileUrl(self, networkType: str):
        labelEncoderFileName = networkType.lower() + '-encoded-labels.json'
        return os.path.join(self.datasets['training']['url'], *['../', labelEncoderFileName])

    def getTrainedModelFileUrl(self, networkType: str):
        modelExportFileName: str = networkType.lower() + '.joblib'
        return os.path.join(self.datasets['training']['url'], *['../', modelExportFileName])

    def readFile(self, url):
        file = open(url, 'r')
        fileContent = file.read()
        file.close()
        return fileContent

    def writeFile(self, url, content: str):
        file = open(url, 'w')
        file.write(content)
        file.close()

    def createFile(self, url, content: str = ''):
        file = open(url, 'w+')
        file.write(content)
        file.close()

    def getLanguagesFolders(self, datasetUrl: str):
        return [d for d in os.scandir(datasetUrl) if d.is_dir()]

    def getExamplesFolders(self, languageFolderUrl: str):
        return [d for d in os.scandir(languageFolderUrl) if d.is_dir()]

    def getExampleFiles(self, exampleFolderUrl: str):
        return [f for f in os.scandir(exampleFolderUrl) if f.is_file()]

    def getRootUrl(self):
        return ROOT_DIR

##


FileManager = FileManagerClass()
