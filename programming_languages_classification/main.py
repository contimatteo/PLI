# /usr/bin/env python3

from solver import ProblemSolver


if __name__ == "__main__":
    # ###################################################
    # ################# INITIALIZATION ##################
    # ###################################################

    print('\n')
    print(' > ')
    Solver = ProblemSolver()

    # initialize
    print(' >  [dataset] initialization')
    Solver.initialize()

    # load the dataset
    print(' >  [dataset] copying files')
    Solver.loadDataset()

    print(' > ')

    # ###################################################
    # #################### TRAINING #####################
    # ###################################################

    # SVM
    print(' >  [training] SVM: algorithm execution')
    Solver.train('SVM')

    # # NaiveBayes
    print(' >  [training] BAYES: algorithm execution')
    # Solver.train('BAYES')

    # # CNN
    print(' >  [training] CNN: algorithm execution')
    # Solver.train('CNN')

    print(' > ')

    # ###################################################
    # ##################### TESTING #####################
    # ###################################################

    # SVM
    print(' >  [testing] SVM: algorithm execution')
    # Solver.test('SVM')

    # # NaiveBayes
    print(' >  [testing] BAYES: algorithm execution')
    # Solver.test('BAYES')

    # # CNN
    print(' >  [testing] CNN: algorithm execution')
    # Solver.test('CNN')

    print(' > ')
    print('\n')

