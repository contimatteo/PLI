# /usr/bin/env python3

from solver import ProblemSolver


if __name__ == "__main__":
    print('')
    Solver = ProblemSolver()

    # ###################################################

    # initialize
    print(' > [setup] Initialization ...')
    Solver.initialize()
    print('')

    # load the dataset
    print(' > [dataset] Copying files ...')
    Solver.loadDataset()
    print('')

    # ###################################################

    # SVM training
    print(' > [training] ==> SVM training execution ...')
    Solver.train('SVM')

    # SVM testing
    print(' > [testing] ==> SVM algorithm execution ...')
    Solver.test('SVM')
    print('')

    # ###################################################

    # NaiveBayes training
    print(' > [training] ==> BAYES training execution ...')
    Solver.train('BAYES')

    # NaiveBayes testing
    print(' > [testing] ==> BAYES algorithm execution ...')
    Solver.test('BAYES')
    print('')

    # ###################################################

    # CNN training
    print(' > [training] ==> CNN training execution ...')
    Solver.train('CNN')

    # CNN testing
    print(' > [testing] ==> CNN algorithm execution ...')
    Solver.test('CNN')
    print('')

