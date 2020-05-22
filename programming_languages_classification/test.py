from keras.models import load_model
import keras.preprocessing.text as kpt
from keras.preprocessing.sequence import pad_sequences
import sys
import os
import json
import numpy as np
from utils import ConfigurationManager, FileManager


##

global dictionary
global model

dictionaryUrl = os.path.join(FileManager.getRootUrl(), 'tmp/wordindex.json')
dictionary = json.loads(FileManager.readFile(dictionaryUrl))

modelUrl = os.path.join(FileManager.getRootUrl(), 'tmp/code_model.h5')
model = load_model(modelUrl)


def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    wordvec = []
    for word in kpt.text_to_word_sequence(text):

        if word in dictionary:
            if dictionary[word] <= 100000:
                wordvec.append([dictionary[word]])
            else:
                wordvec.append([0])
        else:
            wordvec.append([0])

    return wordvec

##


def main():
    data = {"success": False}
    languages = ConfigurationManager.getLanguages()

    matched = 0
    totalExamples = 0

    for languageFolder in FileManager.getLanguagesFolders(FileManager.datasets['testing']['url']):
        language = str(languageFolder.name).lower()
        for exampleFolder in FileManager.getExamplesFolders(languageFolder.path):
            totalExamples += 1

            X_test = []
            originalFileContent = FileManager.readFile(FileManager.getOriginalFileUrl(exampleFolder.path))
            code_snip = originalFileContent
            # print(code_snip, file=sys.stdout)
            word_vec = convert_text_to_index_array(code_snip)
            X_test.append(word_vec)
            X_test = pad_sequences(X_test, maxlen=100)
            # print(X_test[0].reshape(1,X_test.shape[1]), file=sys.stdout)
            y_prob = model.predict(X_test[0].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

            a = np.array(y_prob)
            idx = np.argmax(a)
            if str(languages[idx]) == language:
                matched += 1

            # data["predictions"] = []
            # for i in range(len(languages)):
            #     # print(languages[i], file=sys.stdout)
            #     r = {"label": languages[i], "probability": format(y_prob[i] * 100, '.2f')}
            #     data["predictions"].append(r)

    print('')
    print('')
    print('totalExamples = ' + str(totalExamples))
    print('matched = ' + str(matched))
    print('matched / totalExamples  = ' + str(matched / totalExamples))
    print('')
    print('')


##


if __name__ == "__main__":
    main()
