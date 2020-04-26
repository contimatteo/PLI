# /usr/bin/env python3

import os
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


  def process(self):
      for languageFolder in [f for f in os.scandir(self.DATASET_URI) if f.is_dir()]:
          language = str(languageFolder.name).lower()

          for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:
              example = str(exampleFolder.name).lower()
              # original file ri
              originalFileUri = os.path.join(exampleFolder.path, FILE_NAMES['ORIGINAL'])
              # parsed file uri
              parsedFileName = FILE_NAMES['PARSED']
              parsedFileUri = os.path.join(exampleFolder.path, parsedFileName)              
              # create file, if doesn't exists
              if not os.path.exists(parsedFileUri):
                  # create parsed file content
                  parser = Parser()
                  parser.initialize(originalFileUri, parsedFileUri)
                  parser.parse()
  

  def extract(self):
      for languageFolder in [f for f in os.scandir(self.DATASET_URI) if f.is_dir()]:
          language = str(languageFolder.name).lower()
          # generate the dictionary at 'example' level
          for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:
              example = str(exampleFolder.name).lower()
              # parsed file uri
              parsedFileUri = os.path.join(exampleFolder.path, FILE_NAMES['PARSED'])
              # dictionary file uri
              dictionaryFileName = self.N_NETWORK_PREFIX + "-" + FILE_NAMES['1N_DICTIONARY']
              dictionaryFileUri = os.path.join(exampleFolder.path, dictionaryFileName)
              # replace parsed file content
              dictionaryGenerator = DictionaryGenerator()
              dictionaryGenerator.initialize(parsedFileUri, dictionaryFileUri)
              dictionaryGenerator.generate()
          #
          # TODO: generate the dictionary at 'language' level
          # ...
