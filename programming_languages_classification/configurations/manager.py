# /usr/bin/env python3

import os
import json
import configparser

class ConfigurationManager:
    __languages = None
    __reserved = None

    @staticmethod
    def getLanguages():
        languagesConfigsPath = os.path.join(os.getcwd(), 'configurations/languages.json')

        # Read only once, simulate Swift 'lazy' behaviour.
        if ConfigurationManager.__languages is None:
            with open(languagesConfigsPath) as jsonFile:
                ConfigurationManager.__languages = json.load(jsonFile)

        return ConfigurationManager.__languages


    @staticmethod
    def getReservedWords():
        reservedConfig = os.path.join(os.getcwd(), 'configurations/reserved.json')

        # Read only once, simulate Swift 'lazy' behaviour.
        if ConfigurationManager.__reserved is None:
            with open(reservedConfig) as jsonFile:
                ConfigurationManager.__reserved = json.load(jsonFile)

        return ConfigurationManager.__reserved['words']

    @staticmethod
    def getEscapedTokens():
        return {
            'FILE_BEGIN': '__BOF__\n',
            'FILE_END': '\n__EOF__',
            'NEW_LINE': '__NL__',
            'ALPHA': '__a__',
            'NUMBER': '__n__',
        }

    @staticmethod
    def getFileNames():
        return {
            'ORIGINAL': 'original.txt',
            'PARSED': 'parsed.txt',
            '1N_DICTIONARY': '1n-dictionary.json',
        }

