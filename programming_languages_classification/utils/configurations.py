# /usr/bin/env python3

import os
import json


LANGUAGES_URL = os.path.join(os.getcwd(), 'configurations/languages.json')
RESERVED_WORDS_URL = os.path.join(os.getcwd(), 'configurations/reserved.json')
ESCAPED_TOKENS = {
    'FILE_BEGIN': '__BOF__',
    'FILE_END': '__EOF__',
    'NEW_LINE': '__NL__',
    'ALPHA': '__a__',
    'NUMBER': '__n__',
    'NOT_RELEVANT': '__nr__',
}
CONFIGURATION = {
    'FEATURE_FREQUENCY_THRESHOLD': 0.1,
    'TRAINING_EXAMPLES_NUMBER': 350
}
TOKENIZER_CONFIG = {
    'filter': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
}


class ConfigurationManagerClass:
    reserved_words: list = []
    escaped_tokens : list = []
    languages: dict = {'list': [], 'counters': {}}
    configuration: dict = {}
    tokenizerConfiguration: dict = {}

    def __init__(self):
        self.initialize()

    def initialize(self):
        # languages
        with open(LANGUAGES_URL) as jsonFile:
            self.languages['list'] = json.load(jsonFile)
        # reserved words
        with open(RESERVED_WORDS_URL) as jsonFile:
            self.reserved_words = json.load(jsonFile)
        # escaped tokens
        self.escaped_tokens = ESCAPED_TOKENS
        # config
        self.configuration = CONFIGURATION
        self.tokenizerConfiguration = TOKENIZER_CONFIG

    def getLanguages(self):
        return [str(lang).lower() for lang in self.languages['list']]

    def getReservedWords(self):
        return [str(word).lower() for word in self.reserved_words]

    def getLanguagesExamplesCounter(self):
        return self.languages['counters']

    def setLanguagesExamplesCounter(self, languagesExamplesCounter: dict):
        self.languages['counters'] = languagesExamplesCounter


##

ConfigurationManager = ConfigurationManagerClass()
