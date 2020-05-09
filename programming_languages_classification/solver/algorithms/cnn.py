# /usr/bin/env python3

import os
from .base_tensorflow import _TensorflowAlgorithm
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
from sklearn.model_selection import train_test_split
import os
import json

#

MODEL_CONFIG: dict = {
    'max_features': 100000,
    'embed_dim': 128,
    'lstm_out': 64,
    'batch_size': 32,
    'epochs': 10,
    'max_len_sequences': 100,
    'test_size': 0.5
}


class CNN(_TensorflowAlgorithm):

    def __init__(self):
        super().__init__()
        self.type = 'CNN'
        self.config = MODEL_CONFIG.copy()

    def __prepareModel(self, X, Y):
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

    def __prepareFeatures(self, dataset: str):
        raw_X = []
        raw_Y = []

        sources: dict = self.dataset.getSources(dataset)

        for language in self.dataset.getSources(dataset):
            for exampleDict in sources[language]:
                raw_X.append(exampleDict['original'])
                raw_Y.append(language)

        return raw_X, raw_Y

    def train(self):
        # TODO: fix model file url
        if os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            return self

        # configs
        max_features: int = self.config['max_features']
        batch_size: int = self.config['batch_size']
        epochs: int = self.config['epochs']
        test_size: int = self.config['test_size']

        # preparing features
        codeArchive, languages, = self.__prepareFeatures('training')

        # # get training data
        # X = self.training['X'].tolist()
        # y = self.training['y'].tolist()
        #
        # # train
        # model = svm.SVC()
        # model.fit(X, y)

        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(codeArchive)
        dictionary = tokenizer.word_index
        FileManager.createFile(os.path.join(FileManager.getRootUrl(), 'data/wordindex.json'), json.dumps(dictionary))

        X = tokenizer.texts_to_sequences(codeArchive)
        X = pad_sequences(X, 100)
        Y = pd.get_dummies(languages)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

        # prepare model
        self.__prepareModel(X, Y)

        history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

        # TODO: centralize model export
        # self.exportTrainedModel(model)
        self.model.save(os.path.join(FileManager.getRootUrl(), 'data/code_model.h5'))
        self.model.save_weights(os.path.join(FileManager.getRootUrl(), 'data/code_model_weights.h5'))

        score, acc = self.model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
        print(self.model.metrics_names)
        print("Validation loss: %f" % score)
        print("Validation acc: %f" % acc)


    # def test(self):
    #     if not os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
    #         raise Exception('You can\'t test a model without training it')
    #
    #     # preparing testing data
    #     self.prepareTesting()
    #
    #     Y_Encoder = preprocessing.LabelEncoder()
    #     Y_Encoder.fit(ConfigurationManager.getLanguages())
    #
    #     # get testing data
    #     X = self.testing['X'].tolist() # numpy array
    #     y = self.testing['y'].tolist() # numpy array
    #
    #     # model
    #     model = self.importTrainedModel()
    #
    #     # make predictions
    #     predictions = model.predict(X)
    #
    #     matched = 0
    #     for index, prediction in enumerate(predictions):
    #         predictedLanguage = Y_Encoder.inverse_transform([prediction])[0]
    #         if predictedLanguage == y[index]:
    #             matched += 1
    #
    #     print(' > [testing] ==> ' + self.type + ' (% matched) = ' + str(matched / len(predictions)))



