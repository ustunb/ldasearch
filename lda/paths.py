"""
This file contains:
- Paths to main directories used in the repository
- Functions to get paths for raw datasets and processed datasets
- Functions to get paths for pickle files that contain results
"""

from pathlib import Path

repo_dir = Path(__file__).absolute().parent.parent

data_dir = repo_dir / "data/"
results_dir = repo_dir / "results/"
reports_dir = repo_dir / "reports/"

# create directories if they do not exist
results_dir.mkdir(exist_ok = True)

# path functions
def get_raw_data_file(data_name,  **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return: path for the raw data file used in
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    f = data_dir / data_name / f"{data_name}_data.csv"
    return f

def get_processed_data_file(data_name,  **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    f = data_dir / f"{data_name}_processed.pickle"
    return f

def get_baseline_results_file(data_name, fold_id, baseline_method_name, regime, **kwargs):
    """
    returns file name for pickle files used to store the results of a baseline model training job (e.g., in `train_classifier`)
    :param data_name: string containing name of the dataset
    :param fold_id: string specifying fold_id of cross-validation indices
    :param fold_num: string specifying test fold in cross-validation_indices
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(fold_id, str) and len(fold_id) > 0
    assert isinstance(baseline_method_name, str) and len(baseline_method_name) > 0
    assert isinstance(regime, str) and regime in ["Standard", "Profit", "Prevalence"]
    f = results_dir / f"{data_name}_{fold_id}_{baseline_method_name}_{regime}.results"
    return f


def get_lda_results_file(data_name, fold_id, baseline_method_name, parity_type, groups_to_drop, fnr_slack, fpr_slack, time_limit, regime,  **kwargs):
    """
    returns file name for pickle files used to store the results of a training job (e.g., in `train_classifier`)
    :param data_name: string containing name of the dataset
    :param fold_id: string specifying fold_id of cross-validation indices
    :param fold_num: string specifying test fold in cross-validation_indices
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(fold_id, str) and len(fold_id) > 0
    assert isinstance(baseline_method_name, str) and len(baseline_method_name) > 0
    assert isinstance(parity_type, str) and len(parity_type) > 0
    assert isinstance(groups_to_drop, str) and len(groups_to_drop) > 0
    assert isinstance(fnr_slack, float) and fnr_slack >= 0 and fnr_slack < 1
    assert isinstance(fpr_slack, float) and fpr_slack >= 0 and fpr_slack < 1
    assert isinstance(regime, str) and regime in ["Standard", "Profit", "Prevalence"]
    groups_to_drop = groups_to_drop.replace(",", "")
    fnr_slack = round(fnr_slack * 100)
    fpr_slack = round(fpr_slack * 100)
    f = results_dir / f"{data_name}_{fold_id}_{baseline_method_name}_lda_{parity_type}_Drop{groups_to_drop}_SlackFNR{fnr_slack}_SlackFPR{fpr_slack}_Regime{regime}.results"
    return f

# todo: remove
# def get_prevalence_folds_file(data_name,  **kwargs):
#     """
#     :param data_name: string containing name of the dataset
#     :param kwargs: used to catch other args when unpacking dictionaries
#                    this allows us to call this function as get_results_file_name(**settings)
#     :return: path for the raw data file used in
#     """
#     assert isinstance(data_name, str) and len(data_name) > 0
#     f = data_dir / data_name / f"{data_name}_prevalence_folds.csv"
#     return f
#
# def get_cv_record_file(data_name,  **kwargs):
#     """
#     :param data_name: string containing name of the dataset
#     :param kwargs: used to catch other args when unpacking dictionaries
#                    this allows us to call this function as get_results_file_name(**settings)
#     :return: path for the raw data file used in
#     """
#     assert isinstance(data_name, str) and len(data_name) > 0
#     f = results_dir / f"{data_name}_data_cv_record.csv"
#     return f
#
#
# def get_lda_preds_file(data_name, fold_id, baseline_method_name, parity_type, groups_to_drop, fnr_slack, fpr_slack, time_limit, regime, **kwargs):
#     """
#     returns file name for csv files used to store the predictions of a baseline and lda models
#     :param data_name: string containing name of the dataset
#     :param fold_id: string specifying fold_id of cross-validation indices
#     :param fold_num: string specifying test fold in cross-validation_indices
#     :param kwargs: used to catch other args when unpacking dictionaries
#                    this allows us to call this function as get_results_file_name(**settings)
#     :return:
#     """
#     assert isinstance(data_name, str) and len(data_name) > 0
#     assert isinstance(fold_id, str) and len(fold_id) > 0
#     assert isinstance(baseline_method_name, str) and len(baseline_method_name) > 0
#     assert isinstance(parity_type, str) and len(parity_type) > 0
#     assert isinstance(groups_to_drop, str) and len(groups_to_drop) > 0
#     assert isinstance(fnr_slack, float) and fnr_slack >= 0 and fnr_slack < 1
#     assert isinstance(fpr_slack, float) and fpr_slack >= 0 and fpr_slack < 1
#     assert isinstance(regime, str) and regime in ["Standard", "Profit", "Prevalence"]
#     groups_to_drop = groups_to_drop.replace(",", "")
#     fnr_slack = round(fnr_slack * 100)
#     fpr_slack = round(fpr_slack * 100)
#     f = results_dir / f"{data_name}_{fold_id}_{baseline_method_name}_lda_{parity_type}_Drop{groups_to_drop}_SlackFNR{fnr_slack}_SlackFPR{fpr_slack}_Regime{regime}_preds.csv"
#     return f
#
# def get_lda_stats_file(data_name, fold_id, baseline_method_name, parity_type, groups_to_drop, fnr_slack, fpr_slack, time_limit, regime, **kwargs):
#     """
#     returns file name for csv files used to store the predictions of a baseline and lda models
#     :param data_name: string containing name of the dataset
#     :param fold_id: string specifying fold_id of cross-validation indices
#     :param fold_num: string specifying test fold in cross-validation_indices
#     :param kwargs: used to catch other args when unpacking dictionaries
#                    this allows us to call this function as get_results_file_name(**settings)
#     :return:
#     """
#     assert isinstance(data_name, str) and len(data_name) > 0
#     assert isinstance(fold_id, str) and len(fold_id) > 0
#     assert isinstance(baseline_method_name, str) and len(baseline_method_name) > 0
#     assert isinstance(parity_type, str) and len(parity_type) > 0
#     assert isinstance(groups_to_drop, str) and len(groups_to_drop) > 0
#     assert isinstance(fnr_slack, float) and fnr_slack >= 0 and fnr_slack < 1
#     assert isinstance(fpr_slack, float) and fpr_slack >= 0 and fpr_slack < 1
#     assert isinstance(time_limit, int) and time_limit > 0
#     assert isinstance(regime, str) and regime in ["Standard", "Profit", "Prevalence"]
#     groups_to_drop = groups_to_drop.replace(",", "")
#     fnr_slack = round(fnr_slack * 100)
#     fpr_slack = round(fpr_slack * 100)
#     f = results_dir / f"{data_name}_{fold_id}_{baseline_method_name}_lda_{parity_type}_Drop{groups_to_drop}_SlackFNR{fnr_slack}_SlackFPR{fpr_slack}_Regime{regime}_stats.csv"
#     return f
