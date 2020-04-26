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

