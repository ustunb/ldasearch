from inspect import signature
import numpy as np

from .model import ClassificationModel
from .groups import GroupAttributeEncoder

LINEAR_MODEL_TYPE = ClassificationModel.LINEAR_MODEL_TYPE
DEFAULT_TRAINING_SETTINGS = {

    'logreg': {
        'fit_intercept': True,
        # 'intercept_scaling': 1.0,
        'class_weight': None,
        'penalty': 'none',
        # 'C': 1.0,
        'tol': 1e-4,
        'solver': 'lbfgs',
        'warm_start': False,
        'max_iter': int(1e5),
        'random_state': 2338,
        'verbose': True,
        'n_jobs': 1
        },

    'svm_linear': {
        'fit_intercept': True,
        'intercept_scaling': 1.0,
        'class_weight': None,
        'loss': "hinge",
        'penalty': 'l2',
        'C': 1.0,
        'tol': 1e-4,
        'max_iter': 1e3,
        'dual': True,
        'random_state': None,
        'verbose': False
        },

    'random_forest': {
        'n_estimators': 100,
        'max_features': 'auto',
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'bootstrap': True
        },

    'neural_network': {
        'solver': 'lbfgs',
        'alpha': 1e-5,
        'hidden_layer_sizes': (5, 2),
        'random_state': 2338
        }

    }


def train_sklearn_model(X, G, y, method_name, normalize = False, **settings):

    assert method_name in DEFAULT_TRAINING_SETTINGS.keys(), f'method {method_name} is not supported'

    if G is not None:
        assert 'group_encoding_type' in settings, "group_encoding_type must be specified if G is not None"
        group_encoder = GroupAttributeEncoder(df=G, **settings)
        Z = group_encoder.to_dummies(df=G)
        if X is not None:
            X = np.hstack([Z, X])
        else:
            X = Z
    else:
        group_encoder = None

    # import correct classifier from scikit learn
    model_type = method_name
    if method_name == 'logreg':
        from sklearn.linear_model import LogisticRegression as Classifier
        model_type = LINEAR_MODEL_TYPE
    elif method_name == 'svm_linear':
        from sklearn.linear_model import LinearSVC as Classifier
        model_type = LINEAR_MODEL_TYPE
    elif method_name == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier as Classifier
    elif method_name == 'neural_network':
        from sklearn.neural_network import MLPClassifier as Classifier

    # set missing settings
    settings.update(DEFAULT_TRAINING_SETTINGS.get(method_name))

    # preprocess features
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
        x_shift = np.array(scaler.mean_)
        x_scale = np.sqrt(scaler.var_)
        X = scaler.transform(X)
    else:
        x_shift = np.zeros(X.shape[1], dtype=float)
        x_scale = np.ones(X.shape[1], dtype=float)

    # extract classifier arguments from settings
    clf_args = dict()
    clf_argnames = list(signature(Classifier).parameters.keys())
    for k in clf_argnames:
        if k in settings and settings[k] is not None:
            clf_args[k] = settings[k]

    # fit classifier
    clf = Classifier(**clf_args)
    clf.fit(X, y)

    # pull default parmaters for model object
    predict_handle = clf.predict
    proba_handle = clf.predict_proba
    training_info = dict(settings)
    training_info.update({
        'method_name': method_name,
        'normalize': normalize,
        'x_shift': x_shift,
        'x_scale': x_scale,
        })

    # update model parameters for linear classifiers
    if model_type is LINEAR_MODEL_TYPE:

        b = np.array(clf.intercept_) if settings['fit_intercept'] else 0.0
        w = np.array(clf.coef_)

        # adjust coefficients for unnormalized data
        if normalize:
            w = w * x_scale
            b = b + np.dot(w, x_shift)

        w = np.array(w).flatten()
        b = float(b)

        model_info = {
            'model_type': model_type,
            'intercept': b,
            'coefficients': w,
            'coefficient_idx': range(X.shape[1]),
            }

    model = ClassificationModel(predict_handle=predict_handle,
                                proba_handle=proba_handle,
                                group_encoder=group_encoder,
                                model_info=model_info,
                                training_info=training_info)

    return model
