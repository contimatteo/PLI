# /usr/bin/env python3

from solver import ProblemSolver


if __name__ == "__main__":
    print('\n\n')

    # ###################################################
    # ################# INITIALIZATION ##################
    # ###################################################

    Solver = ProblemSolver()

    # initialize
    print(' > [setup] Initialization ...')
    Solver.initialize()
    print('\n\n')

    # load the dataset
    print(' > [dataset] Copying files ...')
    Solver.loadDataset()
    print('\n\n')

    # ###################################################
    # #################### TRAINING #####################
    # ###################################################

    # # SVM
    # print(' > [train] ==> SVM training execution ...')
    # Solver.train('SVM')
    # print('\n\n')

    # # NaiveBayes
    # print(' > [train] ==> BAYES training execution ...')
    # Solver.train('BAYES')
    # print('\n\n')

    # CNN
    print(' > [test] ==> CNN training execution ...')
    Solver.train('CNN')
    print('\n\n')

    # ###################################################
    # ##################### TESTING #####################
    # ###################################################

    # # SVM
    # print(' > [test] ==> SVM algorithm execution ...')
    # Solver.test('SVM')
    # print('\n\n')

    # # NaiveBayes
    # print(' > [test] ==> BAYES algorithm execution ...')
    # Solver.test('BAYES')
    # print('\n\n')

    # CNN
    print(' > [train] ==> CNN algorithm execution ...')
    Solver.test('CNN')
    print('\n\n')

