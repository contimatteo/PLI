# /usr/bin/env python3

import json
import nltk
from utils import ConfigurationManager


ESCAPED_TOKENS = ConfigurationManager.escaped_tokens


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
        tokens = nltk.word_tokenize(parsedContent)
        tokensCount = len(tokens)
        # generating unique '1N-word' set
        for w in tokens:
            if w != ESCAPED_TOKENS['ALPHA'] and w != ESCAPED_TOKENS['NUMBER'] and w != ESCAPED_TOKENS['FILE_BEGIN'] \
                    and w != ESCAPED_TOKENS['FILE_END'] and w != ESCAPED_TOKENS['NEW_LINE']:
                words.add(w)
        # generating unique '2N-word' set
        for idx, wrd in enumerate(tokens):
            if idx + 1 < tokensCount:
                token = wrd + ' ' + tokens[idx + 1]
                # if ESCAPED_TOKENS['ALPHA'] not in token and ESCAPED_TOKENS['NUMBER'] not in token:
                words.add(token)
        # generating unique '3N-word' set
        for idx, wrd in enumerate(tokens):
            if idx + 2 < tokensCount:
                token = wrd + ' ' + tokens[idx + 1] + ' ' + tokens[idx + 2]
                # if ESCAPED_TOKENS['ALPHA'] not in token and ESCAPED_TOKENS['NUMBER'] not in token:
                words.add(token)
        # write the dictionary
        dictionaryContent: dict = {'words': sorted(words)}
        dictionaryFile.write(json.dumps(dictionaryContent))
        # close files
        parsedFile.close()
        dictionaryFile.close()
