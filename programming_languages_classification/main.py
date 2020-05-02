# /usr/bin/env python3

import nltk
from solver import ProblemSolver

##

nltk.download('punkt')


Solver = ProblemSolver()

##

if __name__ == "__main__":
    # ######################################
    # load the dataset
    print('\n > [setup] Initialization ...')
    Solver.initialize()
    print('\n > [dataset] Copying files ...')
    Solver.loadDataset()
    # ######################################
    # training
    print('')
    Solver.train('SVM')
    Solver.train('BAYES')
    # ######################################
    # testing
    print('')
    Solver.test('SVM')
    Solver.test('BAYES')
    # ######################################
    print('')
