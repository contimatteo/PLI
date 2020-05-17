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
        codeArchive, languages = self.__extractSources('training')
        # self.__prepareFeatures('training')

        # tokenization
        tokenizer = Tokenizer(num_words=max_features, filters=TOKENIZER_CONFIG['filter'])
        tokenizer.fit_on_texts(codeArchive)
        # label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        # (X, Y) creation
        X = tokenizer.texts_to_sequences(codeArchive)
        X = pad_sequences(X, maxlen=input_length)
        Y = Y_Encoder.transform(languages)

        # export words indexes
        self.exportWordsIndexes(tokenizer.word_index)

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
        sourcesToPredict, languages = self.__extractSources('testing')

        # import words indexes
        wordsIndexes = self.importWordsIndexes()
        # label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        # import trained model
        self.importTrainedModel()

        #
        # TESTING
        #

        X = []
        Y_real = Y_Encoder.transform(languages)
        Y_predicted = []

        # preparing Y (languages) label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        for source in sourcesToPredict:
            # tokenization (with unknown tokens)
            word_vec = self.generateWordsIndexesForUnknownExample(wordsIndexes, source)
            # convert 'tokens' to indexes
            X = pad_sequences([word_vec], maxlen=input_length)
            # model prediction
            prediction = self.model.predict(X[0].reshape(1, X.shape[1]))[0]
            Y_predicted.append(prediction)

        accuracy = accuracy_score(Y_real, Y_predicted)
        print(' >  [testing] SVM: algorithm accuracy = ' + str(float("{:.2f}".format(accuracy)) * 100) + '%')

        return self

    ##

    def __prepareModel(self):
        self.model = svm.SVC()
        return self

    def __extractSources(self, dataset: str):
        raw_X = []
        raw_Y = []

        sources: dict = self.dataset.getSources(dataset)

        for language in self.dataset.getSources(dataset):
            for exampleDict in sources[language]:
                raw_X.append(str(exampleDict['original']))
                raw_Y.append(language)

        return raw_X, raw_Y

    def __prepareFeatures(self, dataset: str):
        raw_X = []
        raw_Y = []

        sources: dict = self.dataset.getSources(dataset)

        print('')
        print(self.dataset.countExamples('training'))
        print(self.dataset.countExamples('testing'))
        print('')
        print(self.dataset.getCounters(dataset))
        print('')

        # preparing Y (languages) label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        for language in self.dataset.getSources(dataset):
            for exampleDict in sources[language]:
                raw_X.append(str(exampleDict['original']))
                raw_Y.append(Y_Encoder.transform([language])[0])


        features = []
        numberOfExamples = int(self.dataset.getCounters(dataset))

        for language in self.dataset.getSources(dataset):
            for exampleDict in sources[language]:
                source = str(exampleDict['original'])
                tokens = kpt.text_to_word_sequence(source, filters=TOKENIZER_CONFIG['filter'])

        return raw_X, raw_Y


