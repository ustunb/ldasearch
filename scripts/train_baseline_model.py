"""
This script is used to develop code. It can be called in the Terminal or run in PyCharm
"""
import os
import numpy as np
import sys
from psutil import Process
import argparse
from pathlib import Path
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# this dictionary contains default settings
settings = {
    # dataset
    'data_name': 'adult',
    'groups_to_drop': 'Immigrant,Sex',
    'fold_id': 'K05N01',
    'fold_num_validation': 4,
    'fold_num_test': 5,
    'regime': 'Standard',
    #
    # baseline method
    'baseline_method_name': 'logreg',
    'baseline_load_from_disk': True,
    #
    # General
    'random_seed': 2338,
    }

# parse settings from command line args when script is run from command line
if Process(pid=os.getppid()).name() not in ('pycharm'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default = settings['data_name'], help='dataset name')
    parser.add_argument('--groups_to_drop', type=str, default = settings['groups_to_drop'], help='groups to drop separated by ,')
    parser.add_argument('--fold_id', type=str, default ='K05N01', help='id for cross-validation folds')
    parser.add_argument('--fold_num_test', type=int, default=5, help='test fold number')
    parser.add_argument('--fold_num_validation', type=int, default=4, help='audit fold number')
    parser.add_argument('--regime', type=str, default = settings['regime'], help='Standard, Profit or Prevalance')
    parser.add_argument('--random_seed', type=int, default=109, help='random seed')
    parser.add_argument('--baseline_method_name', type=str, default='logreg', help='method name (logreg)')
    parser.add_argument('--baseline_load_from_disk', type=bool, default=1, help='load baseline from disk')
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# set up path when file is run from the command line
try:
    from lda.paths import repo_dir
except ImportError as e:
    # note that this will fail unless: 1. this file is "in 'lda/scripts/" 2. this file is called from "lda/"
    repo_dir = Path(__file__).absolute().parent.parent
    sys.path.append(str(repo_dir))

###### IMPORTS #####
import dill
from lda.ext.data import BinaryClassificationDataset
from lda.paths import get_processed_data_file, get_baseline_results_file
from lda.ext.training import train_sklearn_model
from lda.utils import compute_group_stats

# load dataset
data = BinaryClassificationDataset.load(file = get_processed_data_file(**settings))
if settings['groups_to_drop'] != ('None'):
    data.drop(settings['groups_to_drop'].split(","))

data.split(fold_id=settings['fold_id'],
           fold_num_validation=settings['fold_num_validation'],
           fold_num_test=settings['fold_num_test'])

X = data.training.X
y = data.training.y
G = data.group_encoder.to_indices(data.training.G)

train_settings = dict(settings)
train_settings.update({
    'method_name': settings['baseline_method_name'],
    'normalize': False,
    'X_names': data.names.X
    })

# train logistic regression model if specified
if settings['baseline_method_name'] == 'logreg':

    train_model = lambda X, G, y: train_sklearn_model(X, G, y, **train_settings)
    baseline_model = train_model(X = X, G = None, y = y)

    if settings['regime'] == "Profit":
        compute_profit = lambda tp, fp: tp - 5.0 * fp
    else:
        compute_profit = lambda tp, fp: tp - 3.0 * fp

    # compute baseline stats
    probs = baseline_model.predict_proba(X)

    # Calculate true positives and false positives at every threshold
    all_fprs, all_tprs, all_thresholds = roc_curve(y, probs)
    n_pos = np.greater(y, 0).sum()
    n_neg = len(y) - n_pos
    all_tps = all_tprs * n_pos
    all_fps = all_fprs * n_neg

    # compute profits at every threshold
    profits = compute_profit(all_tps, all_fps)

    # identify threshold maximizing that maximizes profit
    max_profit_index = np.argmax(profits)
    max_profit_threshold = all_thresholds[max_profit_index]
    intercept_adjustment = np.log(max_profit_threshold / (1.0 - max_profit_threshold))
    adjusted_predictions = np.where(probs >= max_profit_threshold, 1, -1)

    # print changes
    print("threshold to maximize profit", max_profit_threshold)
    print(f"old intercept: {baseline_model.intercept}")
    print(f"adjusted intercept: {baseline_model.intercept + intercept_adjustment}")
    print(f"confusion matrix (current): {confusion_matrix(y, baseline_model.predict(X))}")  # current confusion matrix
    print(f"confusion matrix (desired): {confusion_matrix(y, adjusted_predictions)}")  # confusion matrix that I want

    # adjust the intercept of the logistic regression model
    baseline_model.intercept += intercept_adjustment
    if 0 in baseline_model.predict(X):
        baseline_model.intercept += 1e-10
    print(f"confusion matrix (updated): {confusion_matrix(y, baseline_model.predict(X))}")

    # compute baseline stats
    yhat = baseline_model.predict(X)
    baseline_group_stats = compute_group_stats(yhat, y, G)
    baseline_stats = baseline_group_stats.query('group == -1').to_dict(orient = 'records')[0]

# RF model
if settings['baseline_method_name'] == 'rf':

    # # Defining the parameter grid to search
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'bootstrap': [True, False]
    # }
    #
    # # Creating the Grid Search with Random Forest
    # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    #
    # # Fitting the grid search to the data
    # grid_search.fit(X, y)
    #
    # # Getting the best parameters
    # best_params = grid_search.best_params_
    # best_params
    #
    # optimized_clf = RandomForestClassifier(**best_params)
    # optimized_clf.fit(X, y)
    #
    #
    # y_pred = optimized_clf.predict(data.validation.X)
    # proba_pred = optimized_clf.predict_proba(data.validation.X)[:,1]
    # report = classification_report(data.validation.y, y_pred)
    # print(report)
    # roc_auc_score(data.validation.y, proba_pred)

    # Set class weights corresponding to profit-based thresholds
    if settings['regime'] == "Profit":
        class_weight_ = {
            -1: 5/6,
            1: 1/6
            }
    else:
        class_weight_ = {-1:0.75, 1:0.25}

    # Creating and training the Random Forest Classifier
    baseline_model = RandomForestClassifier(
            n_estimators=100,
            max_depth = 10,
            random_state=303,
            class_weight = class_weight_
            )
    baseline_model.fit(X, y)

baseline_results = dict(train_settings)
baseline_results['model'] = baseline_model
baseline_results_file = get_baseline_results_file(**settings)
with open(baseline_results_file, 'wb') as outfile:
    dill.dump(baseline_results, outfile, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

