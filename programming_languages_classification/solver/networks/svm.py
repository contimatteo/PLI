# /usr/bin/env python3

from sklearn import svm, datasets
import json
from .base import _Network
from sklearn import svm
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


class SvmNetwork(_Network):

    def train(self):
        # TODO: missing training logic ...

        # '{' => 1
        # '}' => 2

        X_Encoder = OrdinalEncoder()
        X = [[1, 0.3], [2, 0.3], [1, 0.2], [2, 0.2]]
        # X_Encoder.fit(X)
        X_encoded = X
        print('\n' + str(X_encoded))

        Y_Encoder = LabelEncoder()
        y = ['js', 'js', 'unknown', 'unknown']
        Y_Encoder.fit(y)
        Y_encoded = Y_Encoder.transform(y)
        print('\n' + str(Y_encoded))

        ##

        model1 = svm.SVC()
        model2 = svm.LinearSVC()
        model3 = svm.SVC(kernel='rbf')
        model4 = svm.SVC(kernel='poly')

        model1.fit(X_encoded, Y_encoded)
        model2.fit(X_encoded, Y_encoded)
        model3.fit(X_encoded, Y_encoded)
        model4.fit(X_encoded, Y_encoded)

        prediction = model1.predict([[1, 0.25]])
        print('\n ' + str(Y_Encoder.inverse_transform(prediction)))

        prediction = model2.predict([[1, 0.25]])
        print('\n ' + str(Y_Encoder.inverse_transform(prediction)))

        prediction = model3.predict([[1, 0.25]])
        print('\n ' + str(Y_Encoder.inverse_transform(prediction)))

        prediction = model4.predict([[1, 0.25]])
        print('\n ' + str(Y_Encoder.inverse_transform(prediction)))
