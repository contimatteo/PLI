# /usr/bin/env python3

import os
import json
from .parser import Parser
from .dictionary import DictionaryGenerator
from configurations import ConfigurationManager

FILE_NAMES: dict = ConfigurationManager.getFileNames()

class FeaturesExtractor:

  def __init__(self):
      self.DATASET_URI: str = "NOT_FOUND"


  def initialize(self, nNetworkPrefix: str, datasetUri: str):
      self.N_NETWORK_PREFIX: str = str(nNetworkPrefix)
      self.DATASET_URI: str = str(datasetUri)


  ##
  
  def __getOriginalFileUri(self, exampleFolder):
    return os.path.join(exampleFolder.path, FILE_NAMES['ORIGINAL'])


  def __getParsedFileUri(self, exampleFolder):
    return os.path.join(exampleFolder.path, FILE_NAMES['PARSED'])   


  def __getDictionaryFileUri(self, exampleFolder):
    dictionaryFileName = self.N_NETWORK_PREFIX + "-" + FILE_NAMES['1N_DICTIONARY']
    return os.path.join(exampleFolder.path, dictionaryFileName)


  def __getFeaturesFileUri(self, languageFolder):
    return os.path.join(languageFolder.path, FILE_NAMES['FEATURES'])


  ##

  def process(self):
      for languageFolder in [f for f in os.scandir(self.DATASET_URI) if f.is_dir()]:
          language = str(languageFolder.name).lower()

          for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:
              # original file ri
              originalFileUri = self.__getOriginalFileUri(exampleFolder)   
              # parsed file uri
              parsedFileUri = self.__getParsedFileUri(exampleFolder)   
              # create file, if doesn't exists
              if not os.path.exists(parsedFileUri):
                  # create parsed file content
                  parser = Parser()
                  parser.initialize(originalFileUri, parsedFileUri)
                  parser.parse()
  

  def extract(self):
      # foreach language
      for languageFolder in [f for f in os.scandir(self.DATASET_URI) if f.is_dir()]:
          examplesCounter = 0 # FIXME: this will be always 400
          wordInExamplesCounter: dict = {}
          wordFrequencies: dict = {}
          
          # check if 'features' file already exists
          featuresFileUri = self.__getFeaturesFileUri(languageFolder)
          if os.path.exists(featuresFileUri):
              continue
          
          # generate the dictionary at 'example' level
          for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:
              examplesCounter += 1
              # parsed file uri
              parsedFileUri = self.__getParsedFileUri(exampleFolder) 
              # dictionary file uri
              dictionaryFileUri = self.__getDictionaryFileUri(exampleFolder)
              # replace parsed file content
              dictionaryGenerator = DictionaryGenerator()
              dictionaryGenerator.initialize(parsedFileUri, dictionaryFileUri)
              dictionaryGenerator.generate()

          # repeating: count '+1' if a word is contained in this example 
          for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:
              # dictionary file uri
              dictionaryFileUri = self.__getDictionaryFileUri(exampleFolder)
              # read dictionary file
              dictionaryFile = open(dictionaryFileUri, 'r')
              dictionaryContent: dict = json.loads(str(dictionaryFile.read()))
              dictionaryFile.close()
              # count all words in this exmaple
              for word in dictionaryContent['words']:
                  if not word in wordInExamplesCounter: wordInExamplesCounter[word] = 1
                  else: wordInExamplesCounter[word] += 1

          # calculating word frequencies foreach language
          for word in wordInExamplesCounter:
              wordFrequencies[word] = wordInExamplesCounter[word] / examplesCounter
          
          # saving the dictionary at 'language' level
          featuresContent: dict = { 'words_frequencies': wordFrequencies }
          featuresFile = open(featuresFileUri, 'w')
          featuresFile.write(json.dumps(featuresContent))
          featuresFile.close()