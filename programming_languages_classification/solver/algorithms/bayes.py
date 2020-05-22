# /usr/bin/env python3

import os
from sklearn import naive_bayes
from ._scikit import _ScikitLearnAlgorithm
from sklearn import preprocessing
from utils import ConfigurationManager, FileManager
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.preprocessing.text as kpt
from collections import Counter
from sklearn.metrics import accuracy_score


ESCAPED_TOKENS = ConfigurationManager.escaped_tokens
TOKENIZER_CONFIG: dict = ConfigurationManager.tokenizerConfiguration
MODEL_CONFIG: dict = {
    'max_features': 100000,
    'max_len_sequences': 100
}


class NaiveBayes(_ScikitLearnAlgorithm):

    def __init__(self):
        super().__init__()
        self.type = 'BAYES'
        self.config = MODEL_CONFIG.copy()

    # def __prepareFeatures(self, dataset: str, encodeLanguagesLabels=True):
    #     raw_X = []
    #     raw_Y = []
    #
    #     sources: dict = self.dataset.getSources(dataset)
    #
    #     # preparing Y (languages) label encoder
    #     Y_Encoder = preprocessing.LabelEncoder()
    #     Y_Encoder.fit(ConfigurationManager.getLanguages())
    #
    #     for language in self.dataset.getSources(dataset):
    #         for exampleDict in sources[language]:
    #             raw_X.append(exampleDict['original'])
    #             if encodeLanguagesLabels:
    #                 raw_Y.append(Y_Encoder.transform([language])[0])
    #             else:
    #                 raw_Y.append(language)
    #
    #     return raw_X, raw_Y

    def train(self):
        if os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            return self

        # max_features: int = self.config['max_features']
        # input_length: int = self.config['max_len_sequences']
        #
        # #
        # # PREPARE FEATURES
        # #
        #
        # # preparing features
        # # codeArchive, languages, = self.__prepareFeatures('training', True)
        # sources, languages = self.__extractSources('training')
        #
        # # tokenization
        # tokenizer = Tokenizer(num_words=max_features)
        # tokenizer.fit_on_texts(sources)
        # # (X, Y) creation
        # X = tokenizer.texts_to_sequences(sources)
        # X = pad_sequences(X, maxlen=input_length)
        # Y = languages
        #
        # # export words indexes
        # self.exportWordsIndexes(tokenizer.word_index)

        #
        # PREPARE FEATURES
        #

        # preparing features
        X, languages = self.__prepareFeatures('training', False)

        # label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        # (X, Y) creation
        Y = Y_Encoder.transform(languages)

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

        # # configs
        # matched = 0
        # totalExamples = 0
        # input_length: int = self.config['max_len_sequences']
        #
        # #
        # # PREPARE FEATURES
        # #
        #
        # # preparing features
        # sourcesToPredict, realLanguages = self.__prepareFeatures('testing', False)
        #
        # # import words indexes
        # wordsIndexes = self.importWordsIndexes()
        # # import trained model
        # self.importTrainedModel()
        #
        # #
        # # TESTING
        # #
        #
        # # preparing Y (languages) label encoder
        # Y_Encoder = preprocessing.LabelEncoder()
        # Y_Encoder.fit(ConfigurationManager.getLanguages())
        #
        # for index, exampleSourceCode in enumerate(sourcesToPredict):
        #     totalExamples += 1
        #
        #     # tokenization
        #     word_vec = self.generateWordsIndexesForUnknownExample(wordsIndexes, exampleSourceCode)
        #     X = pad_sequences([word_vec], maxlen=input_length)
        #     X = X[0].reshape(1, X.shape[1])
        #
        #     # predict
        #     prediction = self.model.predict(X)[0]
        #
        #     # match language prediction
        #     predictedLanguage = Y_Encoder.inverse_transform([prediction])[0]
        #     if predictedLanguage == realLanguages[index]:
        #         matched += 1
        #
        # print('')
        # print(' > [testing] ==> number of total examples = ' + str(totalExamples))
        # print(' > [testing] ==> examples matched = ' + str(matched))
        # print(' > [testing] ==> % success (matched/totalExamples) = ' + str(matched / totalExamples))

        #
        # PREPARE FEATURES
        #

        # preparing features
        X, languages = self.__prepareFeatures('testing', True)

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

        accuracy = accuracy_score(Y_real, Y_predicted)
        print(' >  [testing] BAYES: algorithm accuracy = ' + str(float("{:.2f}".format(accuracy)) * 100) + '%')

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

    def __prepareModel(self):
        self.model = naive_bayes.GaussianNB()
        return self

    def __prepareFeatures(self, dataset: str, importWordsIndexes=False):
        max_features: int = self.config['max_features']
        input_length: int = self.config['max_len_sequences']

        X = []
        Y = []
        sources, languages = self.__extractSources(dataset)

        wordsIndexes = {}
        if not importWordsIndexes:
            # tokenization
            tokenizer = Tokenizer(num_words=max_features)
            tokenizer.fit_on_texts(sources)
            # export words indexes
            self.exportWordsIndexes(tokenizer.word_index)
            # set the indexes
            wordsIndexes = tokenizer.word_index
        else:
            # import and set the words indexes
            wordsIndexes = self.importWordsIndexes()

        # sort the dict by the words indexes
        wordsIndexesSortedByIndex = {k: v for k, v in sorted(wordsIndexes.items(), key=lambda item: item[1])}

        # count the occurencies of each token in the source code.
        for i, source in enumerate(sources):
            language = languages[i]
            features: list = []

            sourceTokens = set(kpt.text_to_word_sequence(source, filters=TOKENIZER_CONFIG['filter']))
            sourceTokensOccurencies = Counter(list(sourceTokens))
            for token, indexValue in wordsIndexesSortedByIndex.items():
                if token not in sourceTokensOccurencies:
                    features.append(0)
                else:
                    features.append(sourceTokensOccurencies[token])
            # X + Y
            X.append(features)
            Y.append(language)

        return X, Y
