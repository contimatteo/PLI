# /usr/bin/env python3

import os
from ._scikit import _ScikitLearnAlgorithm
from sklearn import svm
from sklearn import preprocessing
from utils import ConfigurationManager, FileManager
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


MODEL_CONFIG: dict = {
    'max_features': 100000,
    'max_len_sequences': 100
}


class SVM(_ScikitLearnAlgorithm):

    def __init__(self):
        super().__init__()
        self.type = 'SVM'
        self.config = MODEL_CONFIG.copy()

    def __prepareFeatures(self, dataset: str, encodeLanguagesLabels=True):
        raw_X = []
        raw_Y = []

        sources: dict = self.dataset.getSources(dataset)

        # preparing Y (languages) label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        for language in self.dataset.getSources(dataset):
            for exampleDict in sources[language]:
                raw_X.append(exampleDict['original'])
                if encodeLanguagesLabels:
                    raw_Y.append(Y_Encoder.transform([language])[0])
                else:
                    raw_Y.append(language)

        return raw_X, raw_Y

    def __prepareModel(self):
        self.model = svm.SVC()
        return self

    def train(self):
        if os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            return self

        max_features: int = self.config['max_features']
        input_length: int = self.config['max_len_sequences']

        #
        # PREPARE FEATURES
        #

        # preparing features
        codeArchive, languages, = self.__prepareFeatures('training', True)

        # tokenization
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(codeArchive)
        # (X, Y) creation
        X = tokenizer.texts_to_sequences(codeArchive)
        X = pad_sequences(X, maxlen=input_length)
        Y = languages

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
        matched = 0
        totalExamples = 0
        input_length: int = self.config['max_len_sequences']

        #
        # PREPARE FEATURES
        #

        # preparing features
        sourcesToPredict, realLanguages = self.__prepareFeatures('testing', False)

        # import words indexes
        wordsIndexes = self.importWordsIndexes()
        # import trained model
        self.importTrainedModel()

        #
        # TESTING
        #

        # preparing Y (languages) label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        for index, exampleSourceCode in enumerate(sourcesToPredict):
            totalExamples += 1

            # tokenization
            word_vec = self.generateWordsIndexesForUnknownExample(wordsIndexes, exampleSourceCode)
            X = pad_sequences([word_vec], maxlen=input_length)
            X = X[0].reshape(1, X.shape[1])

            # predict
            prediction = self.model.predict(X)[0]

            # match language prediction
            predictedLanguage = Y_Encoder.inverse_transform([prediction])[0]
            if predictedLanguage == realLanguages[index]:
                matched += 1

        print('')
        print(' > [testing]  ==> number of total examples = ' + str(totalExamples))
        print(' > [testing]  ==> examples matched = ' + str(matched))
        print(' > [testing]  ==> % success (matched/totalExamples) = ' + str(matched / totalExamples))

        return self
