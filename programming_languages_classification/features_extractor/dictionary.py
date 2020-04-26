# /usr/bin/env python3

import json
import nltk
from configurations import ConfigurationManager


ESCAPED_TOKENS = ConfigurationManager.getEscapedTokens()


class DictionaryGenerator:

  def __init__(self):
      self.PARSED_URI: str = "_MISSING_"
      self.DICTIONARY_URI: str = "_MISSING_"


  def initialize(self, parsedUri: str, dictionaryUri: str):
      self.PARSED_URI = parsedUri
      self.DICTIONARY_URI = dictionaryUri


  ##


  def generate(self):
      words: list = [] 
      counters: dict = {}
      # open files
      parsedFile = open(self.PARSED_URI, "r")
      dictionaryFile = open(self.DICTIONARY_URI, "w+")

      # read parsed file contents
      parsedContent = str(parsedFile.read())
      # removed static chars
      parsedContent.replace(ESCAPED_TOKENS['NUMBER'], '') # ISSUE: remove this line for generating a grammar
      # tokenize
      words = [str(x) for x in nltk.word_tokenize(parsedContent)]
      # INFO: unecessary counters logic
      # for word in words:
      #     if not word in counters: counters[word]: int = 1
      #     else: counters[word] += 1

      # write the dictionary
      dictionaryContent: dict = { 'words': words, 'counters': counters }
      dictionaryFile.write(json.dumps(dictionaryContent));
      # close files
      parsedFile.close()
      dictionaryFile.close()
