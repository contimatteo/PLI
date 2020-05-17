# /usr/bin/env python3

import os
from ._scikit import _ScikitLearnAlgorithm
from sklearn import svm
from sklearn import preprocessing
from utils import ConfigurationManager, FileManager
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import keras.preprocessing.text as kpt
from collections import Counter
import math
import json

ESCAPED_TOKENS = ConfigurationManager.escaped_tokens
TOKENIZER_CONFIG: dict = ConfigurationManager.tokenizerConfiguration
MODEL_CONFIG: dict = {
    'max_features': 10000,
    'max_len_sequences': 100
}


class SVM(_ScikitLearnAlgorithm):

    def __init__(self):
        super().__init__()
        self.type = 'SVM'
        self.config = MODEL_CONFIG.copy()

    def train(self):
        if os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            return self

        max_features: int = self.config['max_features']
        input_length: int = self.config['max_len_sequences']

        #
        # PREPARE FEATURES
        #

        # preparing features
        # codeArchive, languages = self.__extractSources('training')
        X, languages = self.__prepareFeatures('training')

        # tokenization
        # tokenizer = Tokenizer(num_words=max_features, filters=TOKENIZER_CONFIG['filter'])
        # tokenizer.fit_on_texts(codeArchive)
        # label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        # (X, Y) creation
        # X = tokenizer.texts_to_sequences(codeArchive)
        # X = pad_sequences(X, maxlen=input_length)
        Y = Y_Encoder.transform(languages)

        # export words indexes
        # self.exportWordsIndexes(tokenizer.word_index)

        #
        # TRAINING
        #

        # prepare model
        self.__prepareModel()
        # training
        self.model.fit(X, Y)
        # export the trained model
        self.exportTrainedModel()

        return self

    def test(self):
        if not os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            raise Exception('You can\'t test a model without training it')

        # configs
        input_length: int = self.config['max_len_sequences']

        #
        # PREPARE FEATURES
        #

        # preparing features
        # sourcesToPredict, languages = self.__extractSources('testing')
        X, languages = self.__prepareFeatures('testing')

        # import words indexes
        # wordsIndexes = self.importWordsIndexes()
        # label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        # import trained model
        self.importTrainedModel()

        #
        # TESTING
        #

        Y_real = Y_Encoder.transform(languages)
        Y_predicted = self.model.predict(X)

        # for source in sourcesToPredict:
        #     # tokenization (with unknown tokens)
        #     word_vec = self.generateWordsIndexesForUnknownExample(wordsIndexes, source)
        #     # convert 'tokens' to indexes
        #     X = pad_sequences([word_vec], maxlen=input_length)
        #     # model prediction
        #     prediction = self.model.predict(X[0].reshape(1, X.shape[1]))[0]
        #     Y_predicted.append(prediction)

        accuracy = accuracy_score(Y_real, Y_predicted)
        print(' >  [testing] SVM: algorithm accuracy = ' + str(float("{:.2f}".format(accuracy)) * 100) + '%')

        return self

    ##

    def __prepareModel(self):
        self.model = svm.SVC()
        return self

    def __extractSources(self, dataset: str):
        X_raw = []
        Y_raw = []
        sources: dict = self.dataset.getSources(dataset)

        for language in self.dataset.getSources(dataset):
            for exampleDict in sources[language]:
                X_raw.append(
                    str(exampleDict['parsed']) \
                        .replace(ESCAPED_TOKENS['ALPHA'], '') \
                        .replace(ESCAPED_TOKENS['NUMBER'], '')
                )
                Y_raw.append(language)

        return X_raw, Y_raw

    def __calculateTokensEntropyLoss(self, dataset: str):
        featuresFileUri = os.path.join(FileManager.getTmpFolderUrl(), 'features/svm.json')
        if os.path.exists(featuresFileUri):
            return self

        sources, languages = self.__extractSources(dataset)
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
        numberOfExamples = self.dataset.countExamples(dataset)

        for language in ConfigurationManager.getLanguages():
            tokensEntropyLoss[language] = {}
            numberOfPositiveExamples: int = self.dataset.getCounters(dataset)[language]
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
            tokensEntropyLoss[language] = {k: v for k, v in sorted(
                tokensEntropyLoss[language].items(), key=lambda item: item[1])
                                           }
            # take first n tokens
            languageFeatures[language] = list(tokensEntropyLoss[language].keys())[:20]

        # export tokens with maximum entropy loss
        FileManager.writeFile(featuresFileUri, json.dumps(languageFeatures))

        return self

    def __prepareFeatures(self, dataset: str):
        # get features file
        self.__calculateTokensEntropyLoss(dataset)
        featuresFileUri = os.path.join(FileManager.getTmpFolderUrl(), 'features/svm.json')
        languageFeatures = json.loads(FileManager.readFile(featuresFileUri))

        X = []
        Y = []
        sources, languages = self.__extractSources(dataset)

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
