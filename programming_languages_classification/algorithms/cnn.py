# /usr/bin/env python3

import numpy as np
from .base import _BaseAlgorithm
from utils import ConfigurationManager, FileManager
from utils import FileManager
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout, Bidirectional
from keras.layers.recurrent import LSTM
import os
import keras.preprocessing.text as kpt
from sklearn.metrics import classification_report


TOKENIZER_CONFIG: dict = ConfigurationManager.tokenizerConfiguration
MODEL_CONFIG: dict = {
    'max_features': 100000,
    'embed_dim': 128,
    'lstm_out': 64,
    'batch_size': 32,
    'epochs': 10,
    'max_len_sequences': 1000
}


class CNN(_BaseAlgorithm):

    def __init__(self):
        super().__init__()
        self.type = 'CNN'
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
        self.__prepareModel(Y)

        # training
        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)

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

    def __prepareModel(self, Y):
        max_features: int = self.config['max_features']
        embed_dim: int = self.config['embed_dim']
        input_length: int = self.config['max_len_sequences']
        lstm_out: int = self.config['lstm_out']

        self.model = Sequential()
        self.model.add(Embedding(max_features, embed_dim, input_length=input_length))
        self.model.add(Conv1D(filters=128, kernel_size=3, padding='same', dilation_rate=1, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(Conv1D(filters=64, kernel_size=3, padding='same', dilation_rate=1, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(lstm_out))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64))
        self.model.add(Dense(len(Y.columns), activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self

    def __prepareFeatures(self, dataset: str, importIndexes=False):
        sources, languages = self.extractSources(dataset)

        # configs
        max_features: int = self.config['max_features']
        input_length: int = self.config['max_len_sequences']

        wordsIndexes: dict = {}
        tokenizer = Tokenizer(num_words=max_features)

        # tokenization
        if not importIndexes:
            tokenizer.fit_on_texts(sources)
            wordsIndexes = tokenizer.word_index
            # export words indexes
            self.exportWordsIndexes(wordsIndexes)
        else:
            wordsIndexes = self.importWordsIndexes()

        # handle unknown words indexes
        if importIndexes:
            wordsIndexesWithUnknownWords: dict = wordsIndexes.copy()
            for index, source in enumerate(sources):
                tokens = kpt.text_to_word_sequence(source, filters=TOKENIZER_CONFIG['filter'])
                for token in tokens:
                    if token not in wordsIndexesWithUnknownWords:
                        wordsIndexesWithUnknownWords[token] = 0
            tokenizer.word_index = wordsIndexesWithUnknownWords

        # X + Y
        X = tokenizer.texts_to_sequences(sources)
        X = pad_sequences(X, maxlen=input_length)
        Y = languages

        return X, Y
