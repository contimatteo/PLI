# /usr/bin/env python3

import os
import shutil
import re as regex
import nltk
nltk.download('punkt')
from configurations import ConfigurationManager


RESERVED_WORDS: list = ConfigurationManager.getReservedWords()
ESCAPED_KEYS = {
  'NUMBER': '__NN__',
  'COMMENT': '__CM__',
}


class Tokenizer:

  def __init__(self):
      self.DATASET_URI: str = "NOT_FOUND"


  def initialize(self, networkPrefix: str, datasetUri: str):
      self.NETWORK_PREFIX: str = str(networkPrefix)
      self.DATASET_URI: str = str(datasetUri)


  ##


  def _isLineAComment(self, line: str):
      # create tokens list
      tokens: list = line.split(' ')
      # check if contains 'reserved words'
      if any(el in RESERVED_WORDS for el in tokens):
          return False
      # remove first occurrency
      tokens.pop(0)
      # check if line contains at least 2 words
      if len(tokens) < 2:
          return False
      # check if contains only words separated by spaces
      return regex.match("^[a-zA-Z ]*$", line)


  def _isWord(self, type: str, w: str):
      if type == 'number':
          return regex.match("^[0-9]*$", w)
      else:
          return False


  def _parseFileContent(self, originalFileUri: str, parsedFileUri: str):
      originalFile = open(originalFileUri, "r")
      parsedFile = open(parsedFileUri, "a")

      for line in originalFile:
          # convert to lower case
          line: str = str(line).lower().replace('\n', '')
          # tokenize
          tokens: list = [str(x) for x in nltk.word_tokenize(line)]
          # analyze each token
          for ix, w in enumerate(tokens):
              # remove single and multiple spaces
              tokens[ix] = regex.sub(' +', ' ', tokens[ix])
              tokens[ix].replace(' ', '')
              # contains only number?
              if self._isWord('number', w): 
                  tokens[ix] = ESCAPED_KEYS['NUMBER']
          # create a line from parsed tokens
          parsedLine: str = ' '.join(x for x in tokens)
          # analyze the entire line
          if self._isLineAComment(parsedLine): # is a comment ?
              print(' > --> comment: '+parsedLine+' | '+str(len(parsedLine.split(' '))), end='\n')
              parsedLine = ESCAPED_KEYS['COMMENT']
          # write the parsed line to the destination file
          parsedFile.write(parsedLine + '\n')

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
              
              # TODO: this is the write logic
              # # create an empty file
              # if not(os.path.exists(parsedFileUri)):
              #     file = open(parsedFileUri, "w+")
              #     file.write('')
              #     file.close()
              
              # TODO: this is the wrong logic
              # delete file, if exists
              if os.path.exists(parsedFileUri):
                  os.remove(parsedFileUri)
              file = open(parsedFileUri, "w+")
              file.write('')
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
