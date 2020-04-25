# /usr/bin/env python3


from solver import ProblemSolver

##

Solver = ProblemSolver()

##

if __name__ == "__main__":
    # load the dataset
    Solver.loadDataset()
    # train
    Solver.train('CCN')
    Solver.train('SVN')
    Solver.train('BAYES')
