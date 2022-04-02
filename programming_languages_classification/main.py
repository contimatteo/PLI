# /usr/bin/env python3

# from solver import ProblemSolver
from algorithms import NaiveBayes, SVM, CNN


if __name__ == "__main__":
    print('\n')
    print(' > ')

    #
    # BAYES

    # initialization
    print(' >  [BAYES]  features initialization ...')
    naiveBayes = NaiveBayes()

    # training
    print(' >  [BAYES]  training ...')
    naiveBayes.train()

    # testing
    print(' >  [BAYES]  testing ...')
    naiveBayes.test()

    print(' > ')

    #
    # SVM

    # initialization
    print(' >  [SVM]  features initialization ...')
    svm = SVM()

    # training
    print(' >  [SVM]  training ...')
    svm.train()

    # testing
    print(' >  [SVM]  testing ...')
    svm.test()

    print(' > ')

    #
    # CNN

    # initialization
    print(' >  [CNN]  features initialization ...')
    cnn: CNN = CNN()

    # training
    print(' >  [CNN]  training ...')
    cnn.train()

    # testing
    print(' >  [CNN]  testing ...')
    cnn.test()

    print(' > ')
    print('\n')

