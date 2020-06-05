# /usr/bin/env python3

import json
from collections import Counter
import re as regex
import numpy as np
from .base import _BaseAlgorithm
from utils import ConfigurationManager, FileManager
from utils import FileManager
import pandas as pd
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import InputLayer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout, Bidirectional, SpatialDropout1D
from keras.layers.recurrent import LSTM
import os
import keras.preprocessing.text as kpt
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer


TOKENIZER_CONFIG: dict = ConfigurationManager.tokenizerConfiguration
MODEL_CONFIG: dict = {
    'max_features': 500000,
    'max_len_sequences': 512,
    'embed_dim': 256,
    'lstm_out': 64,
    'batch_size': 32,
    'epochs': 8,
}


class MG_CNN(_BaseAlgorithm):

    def __init__(self):
        super().__init__()
        self.type = 'MG_CNN'
        self.config = MODEL_CONFIG.copy()

        self.initialize()

    def train(self):
        # check if a trained model already exists
        if os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            return self

        # configs
        batch_size: int = self.config['batch_size']
        epochs: int = self.config['epochs']

        # preparing features
        X, Y = self.__prepareFeatures('training', False)

        # Y dummies encoding
        Y = pd.get_dummies(Y)

        # prepare model
        self.__prepareModel(X, Y)

        # set early stopping monitor so the model stops training when it won't improve anymore
        # early_stopping_monitor = EarlyStopping(monitor='loss', patience=3)
        # training
        # history = self.model.fit(X, Y, batch_size, epochs, verbose=1, callbacks=[early_stopping_monitor])
        history = self.model.fit(X, Y, batch_size, epochs, verbose=1)
        print('\n' + str(self.model.summary()))

        # export the trained model
        self.exportKerasTrainedModel()

        return self

    def test(self):
        if not os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            raise Exception('You can\'t test a model without training it.')

        batch_size: int = self.config['batch_size']

        # preparing features
        X, Y = self.__prepareFeatures('testing', True)

        # import trained model
        self.importKerasTrainedModel()

        # make predictions
        Y_real = pd.get_dummies(Y)
        Y_predicted = self.model.predict(X, batch_size=batch_size)

        target_names = Y_real.columns.tolist()
        Y_real_binary_list = np.argmax(Y_real.values.tolist(), axis=1)
        Y_predicted_binary_list = np.argmax(Y_predicted, axis=1)

        # metrics
        loss, accuracy = self.model.evaluate(X, Y_real, verbose=0, batch_size=batch_size)
        report = classification_report(Y_real_binary_list, Y_predicted_binary_list, target_names=target_names)
        print(' >  [CNN]  classification report exported!')
        print(' >  [CNN]  total accuracy = ' + str(float("{:.2f}".format(accuracy)) * 100) + '%')

        # export the classification report
        self.exportClassificationReport(str(report))

        return self

    def __prepareModel(self, X, Y):
        self.model = Sequential()
        # self.model.add(InputLayer(input_shape=(X.shape[1],)))
        self.model.add(Dense(units=1000, input_shape=(X.shape[1],)))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=800))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=700))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=len(Y.columns), activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self

    # #### #### #### #### #### #### #### #### #### #### #### #### ####

    def extractSources(self, dataset: str):
        X_raw = []
        Y_raw = []
        sources: dict = self.Dataset.getSources(dataset)

        for language in sources:
            for exampleDict in sources[language]:
                source = ''
                for line in str(exampleDict['original']).split('\n'):
                    source += ' '.join(regex.findall(r'[\w\']+|[\[\]!`<>"\'$%&()*+,./:;=#@?\\^{|}~-]+', line)).strip()
                    source += '\n'
                source = ' '.join([w for w in source.split(' ') if len(w.strip()) > 0])
                source = source.replace('\n', ' ')
                #
                X_raw.append(source)
                Y_raw.append(language)

        return X_raw, Y_raw

    def __prepareFeatures(self, dataset: str, importIndexes=False):
        sources, languages = self.extractSources(dataset)

        # tokenization
        if importIndexes:
            vocabulary = self.importVocabulary()
        else:
            OCCURENCE_THRESHOLD_1_GRAM = 0.01
            OCCURENCE_THRESHOLD_2_GRAM = 0.001

            sourcesByLanguage = {}

            for index, source in enumerate(sources):
                language = languages[index]
                if language not in sourcesByLanguage:
                    sourcesByLanguage[language]: list = []
                sourcesByLanguage[language].append(source)

            vocabulary_index = -1
            vocabulary: dict = {}
            tokensOccurrencesByLanguage = {}

            for language in sourcesByLanguage:
                currentSources = sourcesByLanguage[language]
                tokensOccurrencesByLanguage[language] = {}
                vocabulary_for_this_language = set()

                for source in currentSources:
                    tokens = source.split(' ')
                    for tokenIndex in range(0, len(tokens) - 1):
                        token_1gram = str(tokens[tokenIndex])
                        vocabulary_for_this_language.add(token_1gram)
                        if tokenIndex + 1 < len(tokens):
                            token_2gram = str(tokens[tokenIndex] + ' ' + tokens[tokenIndex + 1])
                            vocabulary_for_this_language.add(token_2gram)

                for source in currentSources:
                    for token in vocabulary_for_this_language:
                        if token in source:
                            if token not in tokensOccurrencesByLanguage[language]:
                                tokensOccurrencesByLanguage[language][token] = 0
                            tokensOccurrencesByLanguage[language][token] += 1

                for token in tokensOccurrencesByLanguage[language]:
                    tokenOccurrence = tokensOccurrencesByLanguage[language][token]
                    tokenFreq = tokenOccurrence / len(sources)

                    FREQUENCY_THRESHOLD = 1
                    if len(token.split(' ')) == 1:
                        FREQUENCY_THRESHOLD = OCCURENCE_THRESHOLD_1_GRAM
                    elif len(token.split(' ')) == 2:
                        FREQUENCY_THRESHOLD = OCCURENCE_THRESHOLD_2_GRAM

                    if tokenFreq > FREQUENCY_THRESHOLD:
                        if token not in vocabulary:
                            vocabulary_index += 1
                            vocabulary[token] = vocabulary_index

            # export words indexes
            self.exportVocabulary(vocabulary)

        X = []

        for index, source in enumerate(sources):
            features = []
            Vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), lowercase=False, stop_words=None, vocabulary=vocabulary)
            Vectorizer.fit_transform([source])
            tokensCount = len(Vectorizer.get_feature_names())
            for token in vocabulary:
                occurence = source.count(token)
                if tokensCount > 0:
                    features.append(occurence / tokensCount)
                else:
                    features.append(0)
            X.append(features)

        # X + Y
        # X = pad_sequences(X, maxlen=len(vocabulary.keys()))
        Y = languages

        return np.array(X), np.array(Y)
