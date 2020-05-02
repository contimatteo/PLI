# /usr/bin/env python3

import os
import json


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

        return [str(x).lower() for x in ConfigurationManager.__languages]

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
            'FILE_BEGIN': '__BOF__',
            'FILE_END': '__EOF__',
            'NEW_LINE': '__NL__',
            'ALPHA': '__a__',
            'NUMBER': '__n__',
        }

    @staticmethod
    def getFileNames():
        return {
            'ORIGINAL': 'original.txt',
            'PARSED': 'parsed.txt',
            'FEATURES': 'features.json',
            '3N_DICTIONARY': 'dictionary.json',
        }
