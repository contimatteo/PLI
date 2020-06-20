# /usr/bin/env python3

import dotenv
from algorithms import NaiveBayes, SVM, CNN, NN


dotenv.load_dotenv()
ENABLED_MODELS = {
    'BAYES': True,
    'SVM': True,
    'NN': True,
}


if __name__ == "__main__":
    print('')

    #
    # BAYES
    #

    if ENABLED_MODELS['BAYES']:
        print(' > ')
        # initialization
        print(' >  [BAYES]  features initialization ...')
        naiveBayes = NaiveBayes()

        # training
        print(' >  [BAYES]  training ...')
        naiveBayes.train()

        # testing
        print(' >  [BAYES]  testing ...')
        naiveBayes.test()

    #
    # SVM
    #

    if ENABLED_MODELS['SVM']:
        print(' > ')
        # initialization
        print(' >  [SVM]  features initialization ...')
        svm = SVM()

        # training
        print(' >  [SVM]  training ...')
        svm.train()

        # testing
        print(' >  [SVM]  testing ...')
        svm.test()

    #
    # NN
    #

    if ENABLED_MODELS['NN']:
        print(' > ')
        # initialization
        print(' >  [NN]  features initialization ...')
        nn: CNN = NN()

        # training
        print(' >  [NN]  training ...')
        nn.train()

        # testing
        print(' >  [NN]  testing ...')
        nn.test()

    #

    if ENABLED_MODELS['BAYES'] or ENABLED_MODELS['SVM'] or ENABLED_MODELS['NN']:
        print(' > ')

    print('')

