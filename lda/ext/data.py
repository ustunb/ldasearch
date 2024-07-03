"""
Helper classes to represent and manipulate datasets for a binary classification task
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from copy import copy
import dill
from dataclasses import dataclass, field
from typing import List
from imblearn.over_sampling import RandomOverSampler
from .groups import GroupAttributeSet, GroupAttributeEncoder
from .cv import validate_cvindices

class BinaryClassificationDataset:
    """class to represent/manipulate a dataset for a binary classification task"""

    SAMPLE_TYPES = ('training', 'validation', 'test')

    def __init__(self, X, y, **kwargs):
        """

        :param X:
        :param y:
        :param kwargs:
        """
        self.group_attributes = GroupAttributeSet(pd.DataFrame(kwargs.get('group_df')))
        self.group_encoder = GroupAttributeEncoder(df = self.group_attributes.df, encoding_type = 'intersectional')

        # complete dataset
        self._full = BinaryClassificationSample(parent = self, X = X, y = y)

        # variable names
        self._names = BinaryClassificationVariableNames(parent = self, y = kwargs.get('y_name', 'y'), X = kwargs.get('X_names', ['x%02d' % j for j in range(1, self.d + 1)]))

        # cvindices
        self._cvindices = kwargs.get('cvindices')

        # indicator to check if we have split into train, test, splits
        self.reset()

    def drop(self, names):
        """
        drop one or more group attributes
        :return:
        """
        if isinstance(names, str):
            names = [names]

        assert isinstance(names, list)
        names = [n for n in names if n in self.group_attributes.names]
        if len(names) > 0:
            group_df = pd.DataFrame(self.group_attributes.df).drop(columns = names)
            self.group_attributes = GroupAttributeSet(group_df)
            self.group_encoder = GroupAttributeEncoder(df = self.group_attributes.df, encoding_type = 'intersectional')
            # indicator to check if we have split into train, test, splits
            self.reset()

    def reset(self):
        """
        initialize data object to a state before CV
        :return:
        """
        self._fold_id = None
        self._fold_number_range = []
        self._fold_num_test = 0
        self._fold_num_validation = 0
        self._fold_num_range = 0
        self.training = self._full
        self.validation = self._full.filter(indices = np.zeros(self.n, dtype = np.bool_))
        self.test = self._full.filter(indices = np.zeros(self.n, dtype = np.bool_))
        assert self.__check_rep__()

    #### built-ins ####
    def __check_rep__(self):

        # check complete dataset
        assert self._full.__check_rep__()

        # check names
        assert self.names.__check_rep__()

        # check folds
        if self._cvindices is not None:
            validate_cvindices(self._cvindices)

        if self._fold_id is not None:
            assert self._cvindices is not None

        # check subsamples
        n_total = 0
        for sample_name in self.SAMPLE_TYPES:
            if hasattr(self, sample_name):
                sample = getattr(self, sample_name)
                assert sample.__check_rep__()
                n_total += sample.n

        assert self.n == n_total

        return True

    def __eq__(self, other):
        return (self._full == other._full) and \
               all(np.array_equal(self.cvindices[k], other.cvindices[k]) for k in self.cvindices.keys())

    def __len__(self):
        return self.n

    def __repr__(self):
        return f'ClassificationDataset<n={self.n}, d={self.d}>'

    def __copy__(self):

        cpy = BinaryClassificationDataset(
                X = self.X,
                y = self.y,
                X_names = self.names.X,
                y_name = self.names.y,
                group_df = self.group_attributes.df,
                cvindices = self.cvindices
                )

        return cpy

    #### io functions ####
    @staticmethod
    def read_csv(data_file, **kwargs):
        """
        loads raw data from CSV
        :param data_file: Path to the data_file
        :param helper_file: Path to the helper_file or None.
        :return:
        """

        # extract common file header from dataset file
        file_header = str(data_file).rsplit('_data.csv')[0]

        # convert file names into path objects with the correct extension
        files = {
            'data': f"{file_header}_data",
            'helper': kwargs.get('helper_file', f"{file_header}_helper"),
            #'weights': kwargs.get('weights_file', '{}_weight'.format(file_header)),
            }
        files = {k: Path(v).with_suffix('.csv') for k, v in files.items()}
        assert files['data'].is_file(), f"could not find dataset file: {files['data']}"
        assert files['helper'].is_file(), f"could not find dataset file: {files['helper']}"

        # read helper file
        hf = pd.read_csv(files['helper'], sep= ',')
        hf['is_variable'] = ~(hf['is_outcome'] | hf['is_group_attribute'])
        hf_headers = ['is_outcome', 'is_group_attribute', 'is_variable']
        assert all(hf[hf_headers].isin([0, 1]))
        assert sum(hf['is_outcome']) == 1, 'helper file should specify 1 outcome'
        assert sum(hf['is_variable']) >= 1, 'helper file should specify at least 1 variable'
        if sum(hf['is_group_attribute']) < 1:
            warnings.warn('dataset does not contain group attributes')

        # parse names
        names = {
            'y': hf.query('is_outcome')['header'][0],
            'G': hf.query('is_group_attribute')['header'].tolist(),
            'X': hf.query('is_variable')['header'].tolist(),
            }

        # specify expected data types
        dtypes = {names['y']: np.int_}
        dtypes.update({n: "category" for n in names['G']})
        dtypes.update({n: np.float_ for n in names['X']})

        # read raw data from disk
        df = pd.read_csv(files['data'], sep = ',', dtype = dtypes)
        assert set(df.columns.to_list()) == set(hf['header'].to_list()), 'helper file should contain metadata for every column in the data file'

        # if files['weights'].is_file():
        #     raise NotImplementedError()
        #     w = pd.read_csv(files['weights'], sep=',', header = None).values
        #     w = np.array(w, dtype = np.float).flatten()
        data = BinaryClassificationDataset(
                X = df[names['X']].values,
                y = df[names['y']].replace(0, -1).values,
                group_df = df[names['G']],
                X_names = names['X'],
                y_name = names['y'],
                )

        return data

    def save(self, file, overwrite = False, check_save = True):
        """
        saves object to disk
        :param file:
        :param overwrite:
        :param check_save:
        :return:
        """

        f = Path(file)
        if f.is_file() and overwrite is False:
            raise IOError('file %s already exists on disk' % f)

        # check data integrity
        assert self.__check_rep__()

        # save a copy to disk
        data = copy(self)
        data.reset()
        with open(f, 'wb') as outfile:
            dill.dump({'data': data}, outfile, protocol = dill.HIGHEST_PROTOCOL)

        if check_save:
            loaded_data = self.load(file = f)
            assert data == loaded_data

        return f

    @staticmethod
    def load(file):
        """
        loads processed data file from disk
        :param file: path of the processed data file
        :return: data and cvindices
        """
        f = Path(file)
        if not f.is_file():
            raise IOError(f"file: {f} not found")

        with open(f, 'rb') as infile:
            file_contents = dill.load(infile)
            assert 'data' in file_contents, 'could not find `data` variable in pickle file contents'
            assert file_contents['data'].__check_rep__(), 'loaded `data` has been corrupted'

        data = file_contents['data']
        return data

    #### variable names ####
    @property
    def names(self):
        """ pointer to names of X, y"""
        return self._names

    #### properties of the full dataset ####
    @property
    def n(self):
        """ number of examples in full dataset"""
        return self._full.n

    @property
    def d(self):
        """ number of features in full dataset"""
        return self._full.d

    @property
    def df(self):
        return self._full.df

    @property
    def X(self):
        """ feature matrix """
        return self._full.X

    @property
    def G(self):
        """DataFrame of group attributes"""
        return self._full.G

    @property
    def y(self):
        """ label vector"""
        return self._full.y

    @property
    def classes(self):
        return self._full.classes

    #### cross validation ####
    @property
    def cvindices(self):
        return self._cvindices

    @cvindices.setter
    def cvindices(self, cvindices):
        self._cvindices = validate_cvindices(cvindices)

    @property
    def fold_id(self):
        """string representing the indices of cross-validation folds
        K05N01 = 5-fold CV – 1st replicate
        K05N02 = 5-fold CV – 2nd replicate (in case you want to run 5-fold CV one more time)
        K10N01 = 10-fold CV – 1st replicate
        """
        return self._fold_id

    @fold_id.setter
    def fold_id(self, fold_id):
        assert self._cvindices is not None, 'cannot set fold_id on a BinaryClassificationDataset without cvindices'
        assert isinstance(fold_id, str), 'invalid fold_id'
        assert fold_id in self.cvindices, 'did not find fold_id in cvindices'
        self._fold_id = str(fold_id)
        self._fold_number_range = np.unique(self.folds).tolist()

    @property
    def folds(self):
        """integer array showing the fold number of each sample in the full dataset"""
        return self._cvindices.get(self._fold_id)

    @property
    def fold_number_range(self):
        """range of all possible training folds"""
        return self._fold_number_range

    @property
    def fold_num_validation(self):
        """integer from 1 to K representing the validation fold"""
        return self._fold_num_validation

    @property
    def fold_num_test(self):
        """integer from 1 to K representing the test fold"""
        return self._fold_num_test

    def split(self, fold_id, fold_num_validation = 0, fold_num_test = 0):
        """
        :param fold_id:
        :param fold_num_validation: fold to use as a validation set
        :param fold_num_test: fold to use as a hold-out test set
        :return:
        """
        #
        if fold_id is not None:
            self.fold_id = fold_id
        else:
            assert self.fold_id is not None

        # parse fold numbers
        if fold_num_validation > 0 and fold_num_test > 0:
            assert int(fold_num_test) != int(fold_num_validation)

        if fold_num_validation > 0:
            fold_num_validation = int(fold_num_validation)
            assert fold_num_validation in self._fold_number_range
            self._fold_num_validation = fold_num_validation
        else:
            self._fold_num_validation = 0

        if fold_num_test  > 0:
            fold_num_test = int(fold_num_test)
            assert fold_num_test in self._fold_number_range
            self._fold_num_test = fold_num_test
        else:
            self._fold_num_test = 0

        # update subsamples
        self.training = self._full.filter(indices = np.isin(self.folds, [self.fold_num_validation, self.fold_num_test], invert = True))
        self.validation = self._full.filter(indices = np.isin(self.folds, self.fold_num_validation))
        self.test = self._full.filter(indices = np.isin(self.folds, self.fold_num_test))
        return

@dataclass
class BinaryClassificationSample:
    """class to store and manipulate a subsample of points in a survival dataset"""

    parent: BinaryClassificationDataset
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray = None

    def __post_init__(self):

        self.classes = (-1, 1)
        self.X = np.atleast_2d(np.array(self.X, np.float))
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]

        if self.indices is None:
            self.indices = np.ones(self.n, dtype = np.bool_)
        else:
            self.indices = self.indices.flatten().astype(np.bool_)

        self.update_classes(self.classes)
        assert isinstance(self.G, pd.DataFrame)
        assert self.__check_rep__()

    def __len__(self):
        return self.n

    def __eq__(self, other):
        chk = isinstance(other, BinaryClassificationSample) and np.array_equal(self.y, other.y) and np.array_equal(self.X, other.X)
        return chk

    def __check_rep__(self):
        """returns True is object satisfies representation invariants"""
        assert isinstance(self.X, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        assert self.n == len(self.y)
        assert np.sum(self.indices) == self.n
        assert np.isfinite(self.X).all()
        assert np.isin(self.y, self.classes).all(), 'y values must be stored as {}'.format(self.classes)
        return True

    @property
    def G(self):
        """matrix of group attributes"""
        return self.parent.group_attributes.df[self.indices]

    def update_classes(self, values):
        assert len(values) == 2
        assert values[0] < values[1]
        assert isinstance(values, (np.ndarray, list, tuple))
        self.classes = tuple(np.array(values, dtype = np.int))

        # change y encoding using new classes
        if self.n > 0:
            y = np.array(self.y, dtype = np.float_).flatten()
            neg_idx = np.equal(y, self.classes[0])
            y[neg_idx] = self.classes[0]
            y[~neg_idx] = self.classes[1]
            self.y = y

    @property
    def df(self):
        """
        pandas data.frame containing y, G, X for this sample
        """
        df = pd.DataFrame(self.X, columns = self.parent.names.X)
        df = pd.concat([self.G, df], axis = 1)
        df.insert(column = self.parent.names.y, value = self.y, loc = 0)
        return df

    #### methods #####
    def filter(self, indices):
        """filters samples based on indices"""
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()
        return BinaryClassificationSample(parent = self.parent, X = self.X[indices], y = self.y[indices], indices = indices)

@dataclass
class BinaryClassificationVariableNames:
    """class to represent the names of features, group attributes, and the label in a classification task"""
    parent: BinaryClassificationDataset
    X: List[str] = field(repr = True)
    y: str = field(repr = True, default = 'y')

    def __post_init__(self):
        assert self.__check_rep__()

    @property
    def G(self):
        return self.parent.group_attributes.names

    @staticmethod
    def check_name_str(s):
        """check variable name"""
        return isinstance(s, str) and len(s.strip()) > 0

    def __check_rep__(self):
        """check if this object satisfies representation invariants"""

        assert isinstance(self.X, list) and all([self.check_name_str(n) for n in self.X]), 'X must be a list of strings'
        assert len(self.X) == len(set(self.X)), 'X must be a list of unique strings'

        assert isinstance(self.G, list) and all([self.check_name_str(n) for n in self.G]), 'G must be a list of strings'
        assert len(self.G) == len(set(self.G)), 'G must be a list of unique strings'

        assert self.check_name_str(self.y), 'y must be at least 1 character long'
        return True


def oversample_by_label(data, **kwargs):
    """
    oversample dataset to equalize number of positive and negative labels in each group
    :param data:
    :param kwargs:
    :return:
    """

    group_df = data.group_attributes.df
    group_indices = data.group_encoder.to_indices(group_df)
    ros = RandomOverSampler(**kwargs)

    # generate resampled data
    Xr, yr, Gr = [], [], []
    for k, g in enumerate(data.group_encoder.groups):
        idx = np.isin(group_indices, k)
        Xg, yg = ros.fit_resample(data.X[idx, :], data.y[idx])
        Xr.append(Xg)
        yr.append(yg)
        Gr.append(np.tile(g, (len(yg), 1)))

    # concatenate
    Xr = np.vstack(Xr)
    yr = np.concatenate(yr)
    Gr = pd.DataFrame(np.vstack(Gr), columns=data.G.columns)

    # return new dataset object
    return BinaryClassificationDataset(X=Xr, y=yr, group_df=Gr)


def oversample_by_group_and_label(data, **kwargs):
    """
    oversample dataset to equalize number of positive and negative labels in each group and the size of each group
    :param data:
    :param kwargs:
    :return:
    """
    m = len(data.group_attributes)
    group_df = data.group_attributes.df
    group_indices = data.group_encoder.to_indices(group_df)

    # generate ids for each unique combination (G, y)
    group_values_with_label = np.concatenate((group_indices[:, None], data.y[:, None]), axis=1)
    _, profile_idx = np.unique(group_values_with_label, axis=0, return_inverse=True)

    # oversample groups and labels
    ros = RandomOverSampler(**kwargs)
    D = np.concatenate((data.G, data.X, data.y[:, None]), axis=1)
    D, T = ros.fit_resample(D, profile_idx)
    _, profile_counts = np.unique(T, axis=0, return_counts=True)
    assert np.all(profile_counts == profile_counts[0])

    # split
    X_res = D[:, m:(m + data.d)]
    y_res = D[:, -1]
    G_res = pd.DataFrame(data=D[:, :m], columns=data.G.columns)

    return BinaryClassificationDataset(X=X_res, y=y_res, group_df=G_res)
