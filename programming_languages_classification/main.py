# /usr/bin/env python3

from solver import ProblemSolver


if __name__ == "__main__":
    print('')
    Solver = ProblemSolver()

    # initialize
    print(' > [setup] Initialization ...')
    Solver.initialize()
    print('')

    # load the dataset
    print(' > [dataset] Copying files ...')
    Solver.loadDataset()
    print('')

    # training
    print(' > [training] ==> CNN training execution ...')
    Solver.train('CNN')
    print('')

    # testing
    print(' > [testing] ==> CNN algorithm execution ...')
    Solver.test('CNN')
    print('')

