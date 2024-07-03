"""
This script trains an LDA model
"""
import sys
print(f"Python Version: {sys.version}\n\nPlatform: {sys.platform}")

import os
import numpy as np
import psutil
import argparse
from pathlib import Path

# this dictionary contains default settings
settings = {
    # dataset
    'data_name': 'adult',
    'groups_to_drop': 'Immigrant,Sex',
    'fold_id': 'K05N01',
    'fold_num_validation': 4,
    'fold_num_test': 5,
    'regime': 'Standard',
    ##
    # baseline method
    'baseline_method_name': 'logreg',
    'baseline_sample_type': 'training',
    #
    # LDA parameters
    'sample_type': 'validation',
    'parity_type': 'fnr',
    'fnr_slack': 0.0,
    'fpr_slack': 0.0,
    'lda_load_from_disk': False,
    #
    # training parameters
    'time_limit': 1000,
    'threads': 1,
    'cpx_init_effort': 2,
    'cpx_populate_time': 60,
    #
    # MIP parameters
    'total_l1_norm': 100.0,
    'margin': 0.0001,
    # General
    'random_seed': 109,
    }

# parse settings when the script is run from the command line
ppid = os.getppid()  # get parent process id
process_type = psutil.Process(ppid).name()  # e.g. pycharm, bash
if process_type not in ('pycharm'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default = settings['data_name'], help='dataset name')
    parser.add_argument('--groups_to_drop', type=str, default = settings['groups_to_drop'], help='groups to drop separated by comma')
    parser.add_argument('--fold_id', type=str, default ='K05N01', help='id for cross-validation folds')
    parser.add_argument('--fold_num_test', type=int, default=5, help='test fold number')
    parser.add_argument('--fold_num_validation', type=int, default=4, help='test fold number')
    parser.add_argument('--random_seed', type=int, default=109, help='random seed')
    parser.add_argument('--regime', type=str, default = settings['regime'], help='Standard, Profit or Prevalance')
    parser.add_argument('--baseline_method_name', type=str, default='logreg', help='method name (logreg)')
    parser.add_argument('--baseline_sample_type', type=str, default='training', help='baseline training sample (training, validation, test')
    parser.add_argument('--time_limit', type=int, default=600, help='time limit for cpx')
    parser.add_argument('--sample_type', type=str, default='validation', help='sample type (training, validation, test')
    parser.add_argument('--parity_type', type=str, default='both', help='parity type (both, fnr, fpr')
    parser.add_argument('--fnr_slack', type=float, default=0.00, help='fnr slack')
    parser.add_argument('--fpr_slack', type=float, default=0.00, help='fpr slack')
    parser.add_argument('--lda_load_from_disk', type=bool, default=False, help='load previous LDA files from disk for warmstart')
    parser.add_argument('--cpx_init_effort', type=int, default=2, help='effort level for cpx intialization (1,2,3,4,5)')
    parser.add_argument('--cpx_populate_time', type=int, default=600, help='time limit for cpx populate')
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# if settings['regime'] == "Prevalence":
#   settings['sample_type'] = 'validation'
# set up path when file is run from the command line
# note that this will fail unless: 1. this file is "in 'lda/scripts/" 2. this file is called from "lda/"
try:
    from lda.paths import repo_dir
except ImportError as e:
    repo_dir = Path(__file__).absolute().parent.parent
    sys.path.append(str(repo_dir))

###### IMPORTS #####
import dill
from lda.ext.data import BinaryClassificationDataset
from lda.fitter import LDALinearClassifierFitter
from lda.paths import get_processed_data_file, get_baseline_results_file, get_lda_results_file
from lda.ext.training import train_sklearn_model
from lda.utils import compute_group_stats, apply_prevalence_regime

# load dataset
data = BinaryClassificationDataset.load(file = get_processed_data_file(**settings))

# drop groups
if settings['groups_to_drop'] != ('None'):
    data.drop(settings['groups_to_drop'].split(","))

# split
data.split(fold_id=settings['fold_id'],
           fold_num_validation=settings['fold_num_validation'],
           fold_num_test=settings['fold_num_test'])


# select X, y, G for LDA audit
lda_sample = getattr(data, settings['sample_type'])
X = lda_sample.X
y = lda_sample.y
G = data.group_encoder.to_indices(lda_sample.G)

# upsample to simulate a regime
if settings['regime'] == "Prevalence":
  X, y, G = apply_prevalence_regime(X, y, G)

# load baseline model and compute statistics
baseline_results_file = get_baseline_results_file(**settings)
assert baseline_results_file.exists()
with open(baseline_results_file, 'rb') as infile:
    baseline_results = dill.load(infile)
    baseline_model = baseline_results['model']

# compute baseline statistics
yhat = baseline_model.predict(X)
baseline_group_stats = compute_group_stats(yhat, y, G)
baseline_stats = baseline_group_stats.query('group == -1').to_dict(orient = 'records')[0]

# setup lda fitter
lda_fitter = LDALinearClassifierFitter(
        X = X,
        y = y,
        G = G,
        data = data,
        fnr_baseline = baseline_stats['fnr'],
        fpr_baseline = baseline_stats['fpr'],
        fnr_slack = settings['fnr_slack'],
        fpr_slack = settings['fpr_slack'],
        parity_type = settings['parity_type'],
        sample_type = settings['sample_type'],
        total_l1_norm = settings['total_l1_norm'],
        margin = settings['margin'],
        print_flag = True
        )

#######################################################
# Testing only begins
#######################################################

# compute baseline stats with original intercept
# new_intercept = baseline_model.intercept
# old_intercept = baseline_model.model_info['intercept']
#
# baseline_model.intercept = new_intercept
# yhat = baseline_model.predict(X)
# baseline_group_stats = compute_group_stats(yhat, y, G)
# baseline_stats = baseline_group_stats.query('group == -1').to_dict(orient = 'records')[0]
#
# # setup MIP
# lda_fitter = LDALinearClassifierFitter(
#         data = data,
#         fnr_baseline = baseline_stats['fnr'],
#         fpr_baseline = baseline_stats['fpr'],
#         fnr_slack = 0.1,
#         fpr_slack = 0.1,
#         parity_type = settings['parity_type'],
#         total_l1_norm = 100,
#         margin = settings['margin'],
#         print_flag = True
#         )
# print(baseline_model.intercept)
#######################################################
# Testing only ends
#######################################################

# w = baseline_model.get_parameters(target_l1_norm = lda_fitter.info['total_l1_norm'])
# lda_fitter.fix_coefficients(w)
# lda_fitter.solve()
# lda_fitter.check_solution()

#### Initialization ####

# warmstart using previous solutions
lda_results_file = get_lda_results_file(**settings)
if settings['lda_load_from_disk'] and lda_results_file.exists():
    with open(lda_results_file, 'rb') as infile:
        file_contents = dill.load(infile)
    init_lda_model = file_contents['model']
    lda_fitter.add_initial_solution_from_coefficients(coefs = init_lda_model.get_parameters(), effort_level = settings['cpx_init_effort'])
    # todo: upper and lower bounds using the previous solution

# add coefficients of baseline model for intialiazation if linear
# if baseline_model.model_info['model_type'] == ClassificationModel.LINEAR_MODEL_TYPE:
if settings["baseline_method_name"] == "logreg":
    lda_fitter.add_initial_solution_from_coefficients(coefs = baseline_model.get_parameters(), effort_level = settings['cpx_init_effort'])

# train a logistic regression model using the LDA sample for intialization only
init_settings = dict(settings)
init_settings.update({'method_name': 'logreg', 'normalize': False, 'X_names': data.names.X})
train_linear_model = lambda X, G, y: train_sklearn_model(X, G, y, **init_settings)
init_model = train_linear_model(X=lda_sample.X, G = None, y = lda_sample.y)
lda_fitter.add_initial_solution_from_coefficients(coefs = init_model.get_parameters(), effort_level = settings['cpx_init_effort'])

# todo: change intercept for init_model and add that here too:
all_scores = X.dot(init_model.coefficients) + init_model.intercept
all_intercepts = np.sort(all_scores[1:len(all_scores)])
all_intercepts = np.unique(all_intercepts)
for b in all_intercepts:
    init_model.intercept = b
    coefs = init_model.get_parameters()
    lda_fitter.add_initial_solution_from_coefficients(
            coefs = init_model.get_parameters(),
            effort_level = settings['cpx_init_effort']
            )

# Solve MIP using B&B
lda_fitter.set_parallelization(threads = settings['threads'])
lda_info = lda_fitter.solve(time_limit = settings['time_limit'])

# Populate Solution Pool
if settings['cpx_populate_time'] > 0.0:
    lda_fitter.populate(time_limit = settings['cpx_populate_time'])
    lda_info = lda_fitter.solution_info

# Get classifier
lda_clf = lda_fitter.get_classifier()

# Get solution pool
lda_pool = lda_fitter.solution_pool()

# Construct output of training procedure
results = dict(settings)
results.update({
    'mip': lda_info,
    'fnr_baseline': baseline_stats['fnr'],
    'fpr_baseline': baseline_stats['fpr'],
    'fnr_slack': settings['fnr_slack'],
    'fpr_slack': settings['fpr_slack'],
    'lda_model': lda_clf,
    'lda_pool': lda_pool,
    'lda_intercept': lda_clf.model_info['intercept'],
    'lda_coefficients': lda_clf.model_info['coefficients']
    })

# # save to disk
with open(get_lda_results_file(**settings), 'wb') as outfile:
    dill.dump(results, outfile, protocol=dill.HIGHEST_PROTOCOL, recurse=True)
