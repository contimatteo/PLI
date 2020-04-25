# /usr/bin/env python3

import os
import shutil


class Tokenizer:

  def __init__(self):
      self.DATASET_URI: str = "NOT_FOUND"


  def initialize(self, networkPrefix, datasetUri):
      self.NETWORK_PREFIX: str = str(networkPrefix)
      self.DATASET_URI: str = str(datasetUri)


  ##


  def _parseFileContent(self, originalFileUri, parsedFileUri):
      originalFile = open(originalFileUri, "r")
      parsedFile = open(parsedFileUri, "a")

      for line in originalFile:
        # convert to lower case
        parsedLine = str(line).lower()
        # write the parsed line to the destination file
        parsedFile.write(parsedLine)

      originalFile.close()
      parsedFile.close()
  

  ##


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
              # replace parsed file content
              originalFileUri = os.path.join(exampleFolder.path, 'original.txt')
              self._parseFileContent(originalFileUri, parsedFileUri)
            

  def tokenize(self):
      for languageFolder in [f for f in os.scandir(self.DATASET_URI) if f.is_dir()]:
          language = str(languageFolder.name).lower()

          for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:
              example = str(exampleFolder.name).lower()
              parsedFileName = self.NETWORK_PREFIX + "-dictionary.json"
              parsedFileUri = os.path.join(exampleFolder.path, parsedFileName)
              # create an empty file
              file = open(parsedFileUri, "a+")
              file.write('{}')
              file.close()
