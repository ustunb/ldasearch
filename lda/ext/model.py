import numpy as np
from copy import deepcopy
from inspect import getfullargspec
from lda.ext.groups import GroupAttributeEncoder
from scipy.special import expit
from lda.utils import INTERCEPT_IDX

class ClassificationModel(object):

    LINEAR_MODEL_TYPE = 'linear'
    GENERIC_MODEL_TYPE = 'other'

    def __init__(self, predict_handle, proba_handle, model_info, training_info = None, group_encoder = None):

        if training_info is None:
            training_info = dict()

        # check other fields
        assert isinstance(model_info, dict)
        assert isinstance(training_info, dict)

        # group encoder
        assert group_encoder is None or isinstance(group_encoder, GroupAttributeEncoder)
        self._group_encoder = group_encoder

        self._model_info = deepcopy(model_info)
        self._training_info = deepcopy(training_info)

        self._model_type = model_info.get('model_type', self.GENERIC_MODEL_TYPE)

        if self._model_type is ClassificationModel.LINEAR_MODEL_TYPE:
            self._intercept = float(self._model_info['intercept'])
            self._coefficients = np.array(self._model_info['coefficients']).flatten()
            self._predict_handle = lambda X: np.sign(X.dot(self._coefficients) + self._intercept)
            self._proba_handle = lambda X: expit(X.dot(self._coefficients) + self._intercept)
        else:
            self._intercept = None
            self._coefficients = None
            self._predict_handle = deepcopy(predict_handle)
            self._proba_handle = deepcopy(proba_handle)

        assert self.__check_rep__()

    @property
    def model_info(self):
        return self._model_info

    @property
    def training_info(self):
        return self._training_info

    def predict(self, X, G = None):
        if G is not None and not G.empty:
            assert self._group_encoder is not None
            Z = self._group_encoder.to_dummies(df=G)
            X = Z if X is None else np.hstack([Z, X])
        yhat = self._predict_handle(X).flatten()
        return yhat

    def predict_proba(self, X, G = None):
        if G is not None and not G.empty:
            Z = self._group_encoder.to_dummies(df=G)
            X = Z if X is None else np.hstack([Z, X])
        phat = self._proba_handle(X).flatten()
        return phat

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def intercept(self):
        return self._intercept

    @intercept.setter
    def intercept(self, value):
        value = self._model_info['intercept'] if value is None else value
        b = float(value)
        if np.isfinite(b):
            self._intercept = b
            self._predict_handle = lambda X: np.sign(X.dot(self._coefficients) + self._intercept)
            self._proba_handle = lambda X: expit(X.dot(self._coefficients) + self._intercept)


    def get_parameters(self, target_l1_norm = 1.0):
        """
        returns scaled parameter vector for lienar classifiers
        :param target_l1_norm:
        :return:
        """
        if self._model_type == ClassificationModel.LINEAR_MODEL_TYPE:
            assert np.isfinite(target_l1_norm) and np.greater(target_l1_norm, 0.0)
            w = np.insert(np.array(self.coefficients), INTERCEPT_IDX, self.intercept)
            current_l1_norm = np.linalg.norm(w, ord = 1)
            if not np.isclose(current_l1_norm, target_l1_norm):
                w = (w * target_l1_norm) / current_l1_norm
            return w

    def __repr__(self):
        return f"ClassificationModel<method: {self.model_info}"

    def __check_rep__(self):

        assert isinstance(self.model_info, dict)
        assert isinstance(self.training_info, dict)

        assert callable(self._predict_handle)
        spec = getfullargspec(self._predict_handle)
        assert 'X' in spec.args

        assert callable(self._proba_handle)
        spec_proba = getfullargspec(self._predict_handle)
        assert 'X' in spec_proba.args

        if self._model_type is ClassificationModel.LINEAR_MODEL_TYPE:
            assert np.isfinite(self._intercept)
            assert np.isfinite(self._coefficients).all()
        else:
            assert self._intercept is None
            assert self._coefficients is None
        return True
