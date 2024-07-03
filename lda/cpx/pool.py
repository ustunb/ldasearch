import numpy as np
import pandas as pd

class SolutionPool:
    """
    helper class used to create/manipulate a queue of solutions and objective values
    """

    names = ['solution', 'coefficients', 'objval']

    def __init__(self, df = None):

        if df is None:
            self._df = pd.DataFrame(columns = self.names)
        else:
            assert isinstance(df, pd.DataFrame)
            self._df = df.copy()[self.names]

    def includes(self, solution):
        """
        :param solution: solution vector
        :return: True if there exists another solution in this object that matches th
        """
        return any(solution == old for old in self._df['solution'])

    def add(self, solution, coefficients, objval):
        """
        :param solution:
        :param coefficients:
        :param objval:
        :return:
        """

        if isinstance(objval, (list, np.ndarray)):
            # convert placeholder for prediction constraints to list of appropriate size
            assert all(len(param) == len(objval) for param in (solution, coefficients))
            param_dict = {'solution': solution,
                          'objval': objval,
                          'coefficients': coefficients}
        else:
            param_dict = {'solution': [solution],
                          'objval': [objval],
                          'coefficients': [coefficients]}

        new_df = pd.DataFrame.from_dict(param_dict)
        self._df = pd.concat([self._df, new_df]).reset_index(drop = True)


    def get_best_solution(self):
        best_idx = self._df['objval'].idxmin()
        best_solution = self._df.iloc[best_idx].to_dict()
        return best_solution

    def merge(self, pool):
        self._df.append(pool, sort=False).reset_index(drop=True)

    def get_df(self):
        return self._df.copy(deep=True)

    def clear(self):
        self._df.drop(self._df.index, inplace=True)

    @property
    def size(self):
        return self._df.shape[0]

    @property
    def objvals(self):
        return self._df['objval'].tolist()

    @property
    def solutions(self):
        return self._df['solution'].tolist()

    @property
    def coefficients(self):
        return self._df['coefficients'].tolist()

    def __len__(self):
        return len(self._df)

    def __repr__(self):
        return self._df.__repr__()

    def __str__(self):
        return self._df.__str__()