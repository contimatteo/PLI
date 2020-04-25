# /usr/bin/env python3


from solver import ProblemSolver

##

Solver = ProblemSolver()

##

if __name__ == "__main__":
    # ######################################
    # load the dataset
    print('\n > Initialization ...')
    Solver.initialize()
    print('\n > Copying the dataset files ...')
    Solver.loadDataset()
    # ######################################
    # training
    Solver.train('CCN')
    # ######################################
    # testing
    # Solver.test('CCN')
    # ######################################
    print('')
