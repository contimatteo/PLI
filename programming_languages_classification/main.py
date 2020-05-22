# /usr/bin/env python3

# from solver import ProblemSolver
from algorithms import NaiveBayes


if __name__ == "__main__":
    print('\n')
    print(' > ')

    # ###################################################
    # ##################### BAYES #######################
    # ###################################################

    print(' >  [BAYES]  features initialization ...')
    naiveBayes = NaiveBayes()

    print(' >  [BAYES]  training ...')
    naiveBayes.train()

    print(' >  [BAYES]  testing ...')
    naiveBayes.test()

    print(' > ')

    # ###################################################
    # ####################### SVM #######################
    # ###################################################

    print(' >  [SVM]  features initialization ...')

    print(' >  [SVM]  training ...')

    print(' >  [SVM]  testing ...')

    print(' > ')

    # ###################################################
    # ####################### CNN #######################
    # ###################################################

    print(' >  [CNN]  features initialization ...')

    print(' >  [CNN]  training ...')

    print(' >  [CNN]  testing ...')

    print(' > ')
    print('\n')

