import itertools
import sys
import time
import numpy as np
import pandas as pd

INTERCEPT_NAME = '(Intercept)'
INTERCEPT_IDX = 0

### printing
_LOG_TIME_FORMAT = "%m/%d/%y @ %I:%M %p"

def print_log(msg, print_flag = True):
    if print_flag:
        print_str = f"{time.strftime(_LOG_TIME_FORMAT, time.localtime())} | {msg}"
        print(print_str)
        sys.stdout.flush()

def compute_group_stats(yhat, y, G):

    parity_metrics = ['fnr', 'fpr']

    # reformat inputs
    y = np.array(y, dtype = int).flatten()
    yhat = np.array(yhat, dtype = int).flatten()
    group_labels = np.unique(G)

    # check
    assert np.isin(y, (-1, 1)).all()
    assert np.isin(yhat, (-1, 1)).all()
    assert len(yhat) == len(y)
    assert len(G) == len(y)
    assert np.isin(group_labels, np.arange(0, max(group_labels)+1)).all()

    # compute performance across groups
    df_list = []
    for g in group_labels:
        ix = np.isin(G, g)
        ixp = y[ix] == 1
        ixn = y[ix] == -1
        n_pos = np.sum(ixp)
        n_neg = np.sum(ixn)
        z = np.not_equal(y[ix], yhat[ix])
        df = {
            'group': g,
            #
            'n': n_pos + n_neg,
            'n_pos': n_pos,
            'n_neg': n_neg,
            #
            'mistakes': np.sum(z),
            'fn': np.sum(z[ixp]),
            'fp': np.sum(z[ixn]),
            #
            'tp': n_pos - np.sum(z[ixp]),
            'tn': n_neg - np.sum(z[ixn]),
            }

        df_list.append(df)

    df = pd.DataFrame(df_list)

    # create a total row
    total_df = df.sum(axis = 0).to_frame().transpose()
    total_df['group'] = -1

    # compute rates for all groups and total
    df = pd.concat([df, total_df]).sort_values('group').reset_index(drop = True)
    df['error'] = df['mistakes'] / df['n']
    df['fnr'] = df['fn'] / df['n_pos']
    df['fpr'] = df['fp'] / df['n_neg']
    df['tpr'] = df['tp'] / df['n_pos']
    df['tnr'] = df['tn'] / df['n_neg']

    # split into total and groupwise
    total_df = df.query('group == -1').reset_index(drop = True)
    group_df = df.query('group >= 0').reset_index(drop = True)

    # compute groupwise differences
    abs_disc_metrics = [f'disc_{s}_abs' for s in parity_metrics]
    max_disc_metrics = [f'disc_{s}_max' for s in parity_metrics]
    min_disc_metrics = [f'disc_{s}_min' for s in parity_metrics]
    all_disc_metrics = abs_disc_metrics + max_disc_metrics + min_disc_metrics
    all_disc_metrics.sort()

    # create discrimination gap table
    disc_df = pd.DataFrame(group_df['group'])
    disc_df[all_disc_metrics] = float('nan')

    for a, b in itertools.permutations(group_labels, 2):
        avals = group_df.query(f'group == {a}')[parity_metrics].values
        bvals = group_df.query(f'group == {b}')[parity_metrics].values
        disc_ab = avals - bvals
        idx = disc_df.query(f'group == {a}').index
        abs_diff_a = disc_df.loc[idx, abs_disc_metrics]
        max_diff_a = disc_df.loc[idx, max_disc_metrics]
        min_diff_a = disc_df.loc[idx, min_disc_metrics]
        disc_df.loc[idx, abs_disc_metrics] = np.fmax(np.abs(disc_ab), abs_diff_a)
        disc_df.loc[idx, max_disc_metrics] = np.fmax(disc_ab, max_diff_a)
        disc_df.loc[idx, min_disc_metrics] = np.fmin(disc_ab, min_diff_a)

    # update group
    group_df = pd.merge(left = group_df, right = disc_df)

    # update total
    total_df[abs_disc_metrics] = disc_df.max(axis = 0)[abs_disc_metrics]
    total_df[max_disc_metrics] = disc_df.max(axis = 0)[max_disc_metrics]
    total_df[min_disc_metrics] = disc_df.min(axis = 0)[min_disc_metrics]

    # return a single dataframe
    df = pd.concat([total_df, group_df], axis = 0, ignore_index = True).reset_index(drop = True)
    df = df[['group',  'n', 'n_pos', 'n_neg',  'mistakes', 'fn', 'fp', 'tp', 'tn'] + parity_metrics +  all_disc_metrics]
    return df

def extract_and_compute_stats(df, split, model_type):
    y = df[df.split == split].y
    yhat = df[df.split == split][model_type + "_pred"]
    G = df[df.split == split].G
    stats_df = compute_group_stats(yhat, y, G)
    stats_df['split'] = split
    stats_df['model'] = model_type
    return stats_df

def get_predictions_and_probabilities(model, X):
    """
    Get predictions and probabilities from the model for dataset X.
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return predictions, probabilities

def create_predictions_dataframe(split, X, y, G, model_baseline, model_lda):
    """
    Create a DataFrame with predictions and probabilities for both models,
    along with actual values and group indices.
    """
    base_pred, base_prob = get_predictions_and_probabilities(model_baseline, X)
    lda_pred, lda_prob = get_predictions_and_probabilities(model_lda, X)

    return pd.DataFrame({
        'split': split,
        'y': y,
        'G': G,
        'base_pred': base_pred,
        'base_prob': base_prob,
        'lda_pred': lda_pred,
        'lda_prob': lda_prob
    })

def apply_prevalence_regime(X, y, G):
    """
    Apply the prevalence regime to the given data.

    Args:
    X (array): The feature matrix.
    y (array): The target variable array.
    G (array): The group indices array.

    Returns:
    (array, array, array): The modified X, y, and G arrays after applying the prevalence regime.
    """
    # Find indices where y is 1
    indices_where_one = np.where(y == 1)[0]

    # Randomly selecting half of these indices
    np.random.seed(303)
    np.random.shuffle(indices_where_one)
    selected_indices = indices_where_one[:len(indices_where_one) // 10]

    # Generating the True/False list for filtering
    to_keep = [False if i in selected_indices else True for i in range(len(y))]

    # Filter the arrays
    X_filtered = X[to_keep]
    y_filtered = y[to_keep]
    G_filtered = G[to_keep]

    return X_filtered, y_filtered, G_filtered


