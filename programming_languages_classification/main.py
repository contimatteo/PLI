# /usr/bin/env python3

import nltk
from solver import ProblemSolver

##

nltk.download('punkt')

Solver = ProblemSolver()

##

if __name__ == "__main__":
    print('')

    # initialize
    print(' > [setup] Initialization ...')
    Solver.initialize()
    print('')

    # load the dataset
    print(' > [dataset] Copying files ...')
    Solver.loadDataset()
    print('')

    # training
    Solver.train('SVM')
    # Solver.train('BAYES')
    print('')

    # testing

    Solver.test('SVM')
    # Solver.test('BAYES')
    print('')

