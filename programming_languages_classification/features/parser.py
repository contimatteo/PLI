# /usr/bin/env python3

import re as regex
from utils import ConfigurationManager
from keras.preprocessing.text import text_to_word_sequence


RESERVED_WORDS: list = ConfigurationManager.getReservedWords()
ESCAPED_TOKENS = ConfigurationManager.escaped_tokens
TOKENIZER_CONFIG: dict = ConfigurationManager.tokenizerConfiguration


class Parser:

    def __init__(self):
        self.ORIGINAL_URI: str = "_MISSING_"
        self.PARSED_URI: str = "_MISSING_"

    def initialize(self, originalUri, parserdUri):
        self.ORIGINAL_URI = originalUri
        self.PARSED_URI = parserdUri

    ##

    def _isLineAComment(self, line: str):
        # create tokens list
        words: list = line.split(' ')
        # check if contains 'reserved words'
        if any(el in RESERVED_WORDS for el in words):
            return False
        # remove first occurrency
        words.pop(0)
        # check if line contains at least 3 words
        if len(words) < 3:
            return False
        # check if contains only words separated by spaces
        return regex.match("^[a-zA-Z ]*$", line)

    def _removeMultipleSpacesFrom(self, word: str):
        return str(regex.sub(' +', ' ', word)).replace('\t', '')

    def _splitAlphaAndNumericChars(self, line: str):
        # word tokenization
        words = [str(x) for x in text_to_word_sequence(line, filters=TOKENIZER_CONFIG['filter'])]
        # analyze each word
        for ix, w in enumerate(words):
            # remove single and multiple spaces
            words[ix] = self._removeMultipleSpacesFrom(w)
            words[ix] = words[ix].replace(' ', '')

        return ' '.join(words)

    def _splitPuntaction(self, line: str):
        newLine = regex.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', r' \g<0> ', line).strip()
        return self._removeMultipleSpacesFrom(newLine)

    def _replaceNumericSequence(self, line: str):
        words: list = line.split(' ')
        # analyze each word
        for ix, w in enumerate(words):
            w = w.replace(' ', '')
            if regex.match("^[0-9]+$", w): # this word contains only numbers?
                words[ix] = ESCAPED_TOKENS['NUMBER']

        return ' '.join(words)

    def _replaceAlphaCharacters(self, line: str):
        words: list = line.split(' ')
        # analyze each word
        for ix, w in enumerate(words):
            w = self._removeMultipleSpacesFrom(w)
            if len(w) > 1:
                continue
            if regex.match("^[a-zA-Z]$", w): # this word contains only one letter?
                words[ix] = ESCAPED_TOKENS['ALPHA']

        return ' '.join(words)

    def _replaceEmptyCharacters(self, line: str):
        words: list = []
        # analyze each word
        for ix, w in enumerate(line.split(' ')):
            w = self._removeMultipleSpacesFrom(w)
            if len(w) < 1 or w == '':
                continue
            words.append(w)

        return ' '.join(words)

    ##

    def parse(self):
        parsedContent: str = ""

        # open files
        originalFile = open(self.ORIGINAL_URI, "r")
        parsedFile = open(self.PARSED_URI, "w+")
      
        # PARSING
        # init the parsed file content
        # parsedContent += ESCAPED_TOKENS['FILE_BEGIN'] + ' '
        # parse each line ...
        for line in originalFile:
            parsedLine: str = ""
            # original content to LOWERCASE
            parsedLine += str(line).lower().replace('\n', ' ')  # '\n\n' case removed
            # parse only if this line isn't a COMMENT
            if not self._isLineAComment(parsedLine):
                # split ALPHA and NUMERIC chars
                parsedLine = self._splitAlphaAndNumericChars(parsedLine)
                # split PUNCTUATION chars
                # parsedLine = self._splitPuntaction(parsedLine)
                # replace NUMERIC sequence
                parsedLine = self._replaceNumericSequence(parsedLine)
                # replace ALPHA chars
                parsedLine = self._replaceAlphaCharacters(parsedLine)
                # filter empty chars
                parsedLine = self._replaceEmptyCharacters(parsedLine)
                # replace NEW LINE char
                # parsedLine += ' ' + ESCAPED_TOKENS['NEW_LINE'] + ' '
                # add line to final parsed content
                parsedContent += parsedLine + ' \n '

        # end the parsed file content
        # parsedContent += ' ' + ESCAPED_TOKENS['FILE_END']
        # write parsed file
        parsedFile.write(parsedContent)

        # close files
        originalFile.close()
        parsedFile.close()
