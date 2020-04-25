# /usr/bin/env python3

import os

class Tokenizer:

  def __init__(self):
      self.DATASET_URI: str = "NOT_FOUND"


  def initialize(self, networkPrefix, datasetUri):
      self.NETWORK_PREFIX: str = str(networkPrefix)
      self.DATASET_URI: str = str(datasetUri)


  def parse(self):
      for languageFolder in [f for f in os.scandir(self.DATASET_URI) if f.is_dir()]:
          language = str(languageFolder.name).lower()

          for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:
              example = str(exampleFolder.name).lower()
              parsedFileName = self.NETWORK_PREFIX + "-parsed.txt"
              parsedFileUri = os.path.join(exampleFolder.path, parsedFileName)
              # create an empty file
              file = open(parsedFileUri, "a+")
              file.close()
            

  def tokenize(self):
      for languageFolder in [f for f in os.scandir(self.DATASET_URI) if f.is_dir()]:
          language = str(languageFolder.name).lower()

          for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:
              example = str(exampleFolder.name).lower()
              parsedFileName = self.NETWORK_PREFIX + "-dictionary.json"
              parsedFileUri = os.path.join(exampleFolder.path, parsedFileName)
              # create an empty file
              file = open(parsedFileUri, "a+")
              file.close()