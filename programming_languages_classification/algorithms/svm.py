# /usr/bin/env python3

import os
from .base import _BaseAlgorithm
from sklearn import svm
from sklearn import preprocessing
from utils import ConfigurationManager, FileManager
from sklearn.metrics import accuracy_score, classification_report
import keras.preprocessing.text as kpt
import math
import json


TOKENIZER_CONFIG: dict = ConfigurationManager.tokenizerConfiguration
MODEL_CONFIG: dict = {
    'number_of_tokens_for_language': 20
}


class SVM(_BaseAlgorithm):
    def __init__(self):
        super().__init__()
        self.type = 'SVM'
        self.config = MODEL_CONFIG.copy()

        self.initialize()

    def train(self):
        if os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            return self

        # label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        # preparing features
        X, languages = self.__prepareFeatures('training')

        # (X, Y) creation
        Y = Y_Encoder.transform(languages)

        # prepare model
        self.__prepareModel()

        # training
        self.model.fit(X, Y)

        # export the trained model
        self.exportScikitTrainedModel()

        return self

    def test(self):
        if not os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            raise Exception('You can\'t test a model without training it')

        # label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        # preparing features
        X, languages = self.__prepareFeatures('testing')

        # import trained model
        self.importScikitTrainedModel()

        # make predictions
        Y_real = Y_Encoder.transform(languages)
        Y_predicted = self.model.predict(X)

        # metrics
        accuracy = accuracy_score(Y_real, Y_predicted)
        report = classification_report(Y_real, Y_predicted, target_names=Y_Encoder.classes_)
        print(' >  [SVM]  classification report exported!')
        print(' >  [SVM]  total accuracy = ' + str(float("{:.2f}".format(accuracy)) * 100) + '%')

        # export the classification report
        self.exportClassificationReport(str(report))

        return self

    ##

    def __prepareModel(self):
        self.model = svm.SVC()
        return self

    def __calculateTokensEntropyLoss(self, dataset: str):
        if os.path.exists(FileManager.getFeaturesFileUrl(self.type)):
            return self

        sources, languages = self.extractSources(dataset)
        withTokensOccurencyMap: dict = {}
        withoutTokensOccurencyMap: dict = {}

        for index, source in enumerate(sources):
            language = languages[index]
            tokens = set(kpt.text_to_word_sequence(source, filters=TOKENIZER_CONFIG['filter']))
            for token in tokens:
                if token not in withTokensOccurencyMap:
                    withTokensOccurencyMap[token] = []
                withTokensOccurencyMap[token].append(language)

        for index, source in enumerate(sources):
            language = languages[index]
            tokens = set(kpt.text_to_word_sequence(source, filters=TOKENIZER_CONFIG['filter']))
            for token in withTokensOccurencyMap:
                if token not in tokens:
                    if token not in withoutTokensOccurencyMap:
                        withoutTokensOccurencyMap[token] = []
                    withoutTokensOccurencyMap[token].append(language)

        tokensMetrics: dict = {}

        for language in ConfigurationManager.getLanguages():
            tokensMetrics[language] = {}
            for token in withTokensOccurencyMap:
                tokensMetrics[language][token] = {}
                tokensMetrics[language][token]['numberOfExamplesWithFeatureF']: int = len(withTokensOccurencyMap[token])
                tokensMetrics[language][token]['numberOfExamplesWithoutFeatureF']: int = len(
                    withoutTokensOccurencyMap[token])
                tokensMetrics[language][token]['numberOfPositiveExamplesWithFeatureF']: int = len(
                    [lg for lg in withTokensOccurencyMap[token] if lg == language]
                )
                tokensMetrics[language][token]['numberOfPositiveExamplesWithoutFeatureF']: int = len(
                    [lg for lg in withoutTokensOccurencyMap[token] if lg == language]
                )

        languageFeatures = {}
        tokensEntropyLoss: dict = {}
        numberOfExamples = self.Dataset.countExamples(dataset)
        N_OF_TOKENS_FOR_LANGUAGE: int = self.config['number_of_tokens_for_language']

        for language in ConfigurationManager.getLanguages():
            tokensEntropyLoss[language] = {}
            numberOfPositiveExamples: int = self.Dataset.getCounters(dataset)[language]
            for token in tokensMetrics[language]:
                tokensEntropyLoss[language][token] = 0
                metrics = tokensMetrics[language][token]
                numberOfExamplesWithFeatureF = metrics['numberOfExamplesWithFeatureF']
                numberOfExamplesWithoutFeatureF = metrics['numberOfExamplesWithoutFeatureF']
                numberOfPositiveExamplesWithFeatureF = metrics['numberOfPositiveExamplesWithFeatureF']
                numberOfPositiveExamplesWithoutFeatureF = metrics['numberOfPositiveExamplesWithoutFeatureF']
                # preparing entropy formula vars
                pr_C: float = numberOfPositiveExamples / numberOfExamples
                pr_f: float = numberOfExamplesWithFeatureF / numberOfExamples
                pr_C_f: float = numberOfPositiveExamplesWithFeatureF / numberOfExamplesWithFeatureF
                pr_C_notf: float = numberOfPositiveExamplesWithoutFeatureF / numberOfExamplesWithoutFeatureF

                pr_C = (pr_C if pr_C > 0 else .0001)
                pr_f = (pr_f if pr_f > 0 else .0001)
                pr_C_f = (pr_C_f if pr_C_f > 0 else .0001)
                pr_C_notf = (pr_C_notf if pr_C_notf > 0 else .0001)

                pr_C = (pr_C if pr_C < 1 else .9999)
                pr_f = (pr_f if pr_f < 1 else .9999)
                pr_C_f = (pr_C_f if pr_C_f < 1 else .9999)
                pr_C_notf = (pr_C_notf if pr_C_notf < 1 else .9999)

                # calculating token's entropy
                e = -(pr_C * math.log2(pr_C)) - ((1 - pr_C) * math.log2(1 - pr_C))
                e_f = -(pr_C_f * math.log2(pr_C_f)) - ((1 - pr_C_f) * math.log2(1 - pr_C_f))
                e_not_f = -(pr_C_notf * math.log2(pr_C_notf)) - ((1 - pr_C_notf) * math.log2(1 - pr_C_notf))
                tokensEntropyLoss[language][token] = e - (e_f * pr_f) + (e_not_f * (1 - pr_f))

            # sort entropy values by desc order
            tokensEntropyLoss[language] = {
                k: v for k, v in sorted(tokensEntropyLoss[language].items(), key=lambda item: item[1])
            }
            # take first n tokens
            languageFeatures[language] = list(tokensEntropyLoss[language].keys())[:N_OF_TOKENS_FOR_LANGUAGE]

        # export tokens with maximum entropy loss
        FileManager.writeFile(FileManager.getFeaturesFileUrl(self.type), json.dumps(languageFeatures))

        return self

    def __prepareFeatures(self, dataset: str):
        # find or create the features file
        self.__calculateTokensEntropyLoss(dataset)
        # get features file
        languageFeatures = json.loads(FileManager.readFile(FileManager.getFeaturesFileUrl(self.type)))

        X = []
        Y = []
        sources, languages = self.extractSources(dataset)

        for idx, source in enumerate(sources):
            language = languages[idx]
            features = []
            tokens = set(kpt.text_to_word_sequence(source, filters=TOKENIZER_CONFIG['filter']))
            # X
            for _lang in languageFeatures:
                for _tk in languageFeatures[_lang]:
                    if _tk in tokens: features.append(1)
                    else: features.append(0)
            X.append(features)
            # Y
            Y.append(language)

        return X, Y
