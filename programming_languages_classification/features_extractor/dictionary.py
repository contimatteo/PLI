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
      words: set = set([])
      # open files
      parsedFile = open(self.PARSED_URI, "r")
      dictionaryFile = open(self.DICTIONARY_URI, "w+")

      # read parsed file contents
      parsedContent = str(parsedFile.read())
      # generating unique 'words' set
      # words = [str(x) for x in nltk.word_tokenize(parsedContent)]
      for w in nltk.word_tokenize(parsedContent):
          # ISSUE: remove the next line for generating a grammar
          if w != ESCAPED_TOKENS['ALPHA'] and w != ESCAPED_TOKENS['NUMBER']:
              words.add(w)
      # write the dictionary
      dictionaryContent: dict = { 'words': sorted(words) }
      dictionaryFile.write(json.dumps(dictionaryContent));
      # close files
      parsedFile.close()
      dictionaryFile.close()
