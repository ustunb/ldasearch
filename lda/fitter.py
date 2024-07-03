import numpy as np
import itertools
import warnings
from scipy.special import expit
from cplex import Cplex, SparsePair
from .ext.model import ClassificationModel
from .utils import INTERCEPT_IDX, INTERCEPT_NAME
from .cpx.pool import SolutionPool
from .cpx.utils import add_variable, set_mip_time_limit, StatsCallback, get_mip_stats, \
    has_solution, add_mip_start, set_mip_node_limit

def check_mip_solution(mip, mip_info, mip_data, debug_flag = False):

    debugger = lambda: True
    if debug_flag:
        from dev.debug import ipsh as debugger

    # check solutions
    sol = mip.solution
    names = mip_info['names']
    objval = sol.get_objective_value()
    theta_pos = np.array(sol.get_values(names['theta_pos']))
    theta_neg = np.array(sol.get_values(names['theta_neg']))
    theta_sign = np.array(sol.get_values(names['theta_sign']))
    mistakes_pos = np.array(sol.get_values(names['mistakes_pos']))
    mistakes_neg = np.array(sol.get_values(names['mistakes_neg']))
    theta = theta_pos + theta_neg

    try:
        # check coefficients
        assert set(np.flatnonzero(theta_pos)).isdisjoint(set(np.flatnonzero(theta_neg)))
        assert np.greater_equal(theta_pos, 0.0).all()
        assert np.less_equal(theta_neg, 0.0).all()
        assert np.isclose(mip_info['total_l1_norm'], np.linalg.norm(theta, ord = 1))
        assert np.isclose(theta_sign[theta > 0.0], 1.0).all()
        assert np.isclose(theta_sign[theta < 0.0], 0.0).all()
    except AssertionError as e:
        print('error in coefficient variable behavior')
        debugger()

    # check that mistakes are actually miskakes
    y_pos = mip_data['Y'][mip_data['x_to_u_pos_idx']]
    y_neg = mip_data['Y'][mip_data['x_to_u_neg_idx']]
    assert np.all(y_pos == 1) and np.all(y_neg == -1)

    # check mistake combination
    margin = abs(mip_info['margin'])

    S_pos = mip_data['U_pos'].dot(theta)
    S_neg = mip_data['U_neg'].dot(theta)
    yhat_pos = np.sign(S_pos)

    # positive mistakes
    expected_mistakes_pos = np.not_equal(yhat_pos, y_pos)
    bug_idx = np.flatnonzero(np.not_equal(mistakes_pos, expected_mistakes_pos))
    if any(bug_idx):
        warnings.warn("found %1.0f cases where mistakes_neg[i] may be on margin" % np.sum(bug_idx))

    bug_idx = set(bug_idx) - set(np.flatnonzero(np.isclose(S_pos, margin)))
    try:
        assert len(bug_idx) == 0
    except AssertionError as e:
        print('error with mistakes_pos', e)
        debugger()

    # negative mistakes
    yhat_neg = np.sign(S_neg)
    expected_mistakes_neg = np.not_equal(yhat_neg, y_neg)
    bug_idx = np.flatnonzero(np.not_equal(mistakes_neg, expected_mistakes_neg))
    if any(bug_idx):
        warnings.warn("found %1.0f cases where mistakes_pos[i] may be on margin" % np.sum(bug_idx))

    bug_idx = set(bug_idx) - set(np.flatnonzero(np.isclose(S_neg, -margin)))
    try:
        assert len(bug_idx) == 0
    except AssertionError as e:
        print('error with mistakes_neg', e)
        debugger()

    # todo: check TPR exceeds bounds
    # todo: check FPR exceeds bounds
    # todo: check TPR per group is computed correctly
    # todo: check FPR per group is computed correctly
    # todo: check that TPR discrimination is computed correctly
    # todo: check that FPR discrimination is computed correctly
    # todo: check that overall discrimination is computed correctly

    #
    # try:
    #     # check conflicted points
    #     conflicted_pairs = tuple(data['conflicted_pairs'])
    #     conflicted_mistakes = np.array([mistakes_pos[pi] + mistakes_neg[qi] for pi, qi in conflicted_pairs])
    #     assert np.all(conflicted_mistakes >= 1)
    #     if np.any(conflicted_mistakes >= 2):
    #         warnings.warn("BUG: found %d cases where z[i] + z[i'] >= 2 on conflicting points" % np.sum(conflicted_mistakes >= 2))
    #
    #     # check lower bound on mistakes
    #     n_equivalent_points = sum([min(data['n_counts_pos'][pi], data['n_counts_neg'][qi]) for pi, qi in conflicted_pairs])
    #     assert np.greater_equal(objval, n_equivalent_points)
    #
    # except AssertionError as e:
    #     print('error with # of equivalent points', e)
    #     debugger()

    return True

def to_mip_data(X, y, G, data, sample_type = 'training'):

    n_samples, n_variables = X.shape
    # X_shift = np.zeros(n_variables)
    # X_scale = np.ones(n_variables)

    # groups
    group_names = data.group_encoder.dummy_names
    group_indexer = data.group_encoder.indexer
    group_labels = np.unique(G)
    n_groups = len(group_names)
    group_pairs = list(itertools.permutations(group_labels, 2))

    # add intercept
    X = np.insert(X, INTERCEPT_IDX, axis = 1, values = np.ones(n_samples))
    coefficient_names = [INTERCEPT_NAME] + data.names.X
    coefficient_idx = [i for i, n in enumerate(coefficient_names) if n is not INTERCEPT_NAME]
    intercept_idx = INTERCEPT_IDX

    # compression
    pos_ind = y == 1
    neg_ind = ~pos_ind
    n_samples_pos = np.sum(pos_ind)
    n_samples_neg = n_samples - n_samples_pos

    GX = np.hstack((G[:, None], X))
    X_pos = GX[pos_ind, ]
    X_neg = GX[neg_ind, ]
    U_pos, x_pos_to_u_pos_idx, u_pos_to_x_pos_idx, n_counts_pos = np.unique(X_pos, axis = 0, return_index = True, return_inverse = True, return_counts = True)
    U_neg, x_neg_to_u_neg_idx, u_neg_to_x_neg_idx, n_counts_neg = np.unique(X_neg, axis = 0, return_index = True, return_inverse = True, return_counts = True)
    n_points_pos = U_pos.shape[0]
    n_points_neg = U_neg.shape[0]
    x_to_u_pos_idx = np.flatnonzero(pos_ind)[x_pos_to_u_pos_idx]
    x_to_u_neg_idx = np.flatnonzero(neg_ind)[x_neg_to_u_neg_idx]

    # check compression
    assert np.all(y[x_to_u_pos_idx,] == 1)
    assert np.all(y[x_to_u_neg_idx,] == -1)
    assert np.all(X[x_to_u_pos_idx,] == U_pos[:, 1:])
    assert np.all(X[x_to_u_neg_idx,] == U_neg[:, 1:])
    assert np.all(G[x_to_u_pos_idx,] == U_pos[:, 0])
    assert np.all(G[x_to_u_neg_idx,] == U_neg[:, 0])

    # split G from U
    G_pos = U_pos[:, 0]
    G_neg = U_neg[:, 0]
    U_pos = np.array(U_pos[:, 1:], dtype = np.float_)
    U_neg = np.array(U_neg[:, 1:], dtype = np.float_)

    # group parameters
    assert n_groups == len(group_labels)
    group_indices = {
        'pos': {g: np.flatnonzero(np.isin(G_pos, g)) for g in group_labels},
        'neg': {g: np.flatnonzero(np.isin(G_neg, g)) for g in group_labels}
        }

    # check group counts
    group_indices_pos = list(itertools.chain.from_iterable(group_indices['pos'].values()))
    group_indices_neg = list(itertools.chain.from_iterable(group_indices['neg'].values()))
    assert set(group_indices_pos) == set(range(n_points_pos))
    assert set(group_indices_neg) == set(range(n_points_neg))
    n_group_counts_pos = {g: np.sum(n_counts_pos[idx]) for g, idx in group_indices['pos'].items()}
    n_group_counts_neg = {g: np.sum(n_counts_neg[idx]) for g, idx in group_indices['neg'].items()}

    mip_data = {
        #
        'format': 'mip',
        #
        'sample_type': sample_type,
        'variable_names': coefficient_names,
        'intercept_idx': intercept_idx,
        'coefficient_idx': coefficient_idx,
        'n_variables': n_variables,
        #
        # data points
        'U_pos': U_pos,
        'U_neg': U_neg,
        'conflicted_pairs': [],#get_common_row_indices(U_pos, U_neg),
        #
        # counts
        'n_samples': n_samples,
        'n_samples_pos': n_samples_pos,
        'n_samples_neg': n_samples_neg,
        'n_counts_pos': n_counts_pos,
        'n_counts_neg': n_counts_neg,
        #
        # group parameters
        'n_groups': n_groups,
        'group_indices': group_indices,
        'G_pos': G_pos,
        'G_neg': G_neg,
        'n_group_counts_pos': n_group_counts_pos,
        'n_group_counts_neg': n_group_counts_neg,
        'group_pairs': group_pairs,
        'group_indexer': group_indexer,
        'group_labels': group_labels,
        #
        # debugging parameters
        'Y': y,
        'x_to_u_pos_idx': x_to_u_pos_idx,
        'x_to_u_neg_idx': x_to_u_neg_idx,
        'u_pos_to_x_pos_idx': u_pos_to_x_pos_idx,
        'u_neg_to_x_neg_idx': u_neg_to_x_neg_idx,
        'n_points_pos': n_points_pos,
        'n_points_neg': n_points_neg,
        }

    return mip_data

def build_lda_mip(mip_data, var_name_fmt, **kwargs):
    """
    :param data:
    :param settings:
    :param var_name_fmt:
    :return:
    --
    variable vector = [theta_pos, theta_neg, sign, mistakes_pos, mistakes_neg, loss_pos, loss_neg] ---
    --

    ----------------------------------------------------------------------------------------------------------------
    name                  length              type        description
    ----------------------------------------------------------------------------------------------------------------
    theta_pos:            d x 1               real        positive components of weight vector
    theta_neg:            d x 1               real        negative components of weight vector
    theta_sign:           d x 1               binary      sign of weight vector. theta_sign[j] = 1 -> theta_pos > 0; theta_sign[j] = 0  -> theta_neg = 0.
    mistakes_pos:         n_points_pos x 1    binary      mistake_pos[i] = 1 if mistake on instance i
    mistakes_neg:         n_points_neg x 1    binary      mistake_neg[i] = 1 if mistake on instance i
    fn:                   n_groups x 1        int         fn[g] = false negatives for group g = mistakes on positive labels for group g
    fp:                   n_groups x 1        int         fp[g] = false positives for group g = mistakes on positive labels for group g
    fn_total:             1                   int         fn_total = sum of all false positives
    fp_total:             1                   int         fn_total = sum of all false negative
    disc_fnr:             n_groups choose 2   real        disc_fpr[g][g'] = difference in FPR between groups g and g'
    disc_fpr:             n_groups choose 2   real        disc_fnr[g][g'] = difference in FPR between groups g and g'
    disc_total            1                   real        total_discrimination between groups
    """
    # parse settings
    settings = dict(LDALinearClassifierFitter.SETTINGS)
    settings.update(kwargs)

    # check mip data
    assert mip_data['format'] == 'mip'

    # basic mip parameters
    total_l1_norm = np.abs(settings['total_l1_norm']).astype(float)
    assert np.greater(total_l1_norm, 0.0)
    margin = np.abs(settings['margin']).astype(float)
    assert np.isfinite(margin)

    # lengths
    n_variables = len(mip_data['variable_names'])
    n_points_pos, n_points_neg = mip_data['n_points_pos'], mip_data['n_points_neg']
    n_groups = mip_data['n_groups']

    # coefficient bounds
    theta_ub = np.repeat(total_l1_norm, n_variables)
    theta_lb = np.repeat(-total_l1_norm, n_variables)

    # fnr requirement
    assert np.greater_equal(settings['fnr_baseline'], 0.0) and np.less_equal(settings['fnr_baseline'], 1.0)
    assert np.greater_equal(settings['fnr_slack'], 0.0) and np.less_equal(1.0 - settings['fnr_slack'], 1.0)
    max_fn_total = np.ceil(mip_data['n_samples_pos'] * (settings['fnr_baseline'] + settings['fnr_slack']))
    max_fn_total = np.minimum(max_fn_total, mip_data['n_samples_pos'])

    # fpr requirement
    assert np.greater_equal(settings['fpr_baseline'], 0.0) and np.less_equal(settings['fpr_baseline'], 1.0)
    assert np.greater_equal(settings['fpr_slack'], 0.0) and np.less_equal(1.0 - settings['fpr_slack'], 1.0)
    max_fp_total = np.ceil(mip_data['n_samples_neg'] * (settings['fpr_baseline'] + settings['fpr_slack']))
    max_fp_total = np.minimum(max_fp_total, mip_data['n_samples_neg'])

    # build mip
    mip = Cplex()
    mip.objective.set_sense(mip.objective.sense.minimize)
    vars = mip.variables
    cons = mip.linear_constraints

    # define variables
    print_vnames = lambda vfmt, vcnt: list(map(lambda v: vfmt % v, range(vcnt)))
    names = {
        'theta_pos': print_vnames(var_name_fmt['theta_pos'], n_variables),
        'theta_neg': print_vnames(var_name_fmt['theta_neg'], n_variables),
        'theta_sign': print_vnames(var_name_fmt['theta_sign'], n_variables),
        'mistakes_pos': print_vnames(var_name_fmt['mistake_pos'], n_points_pos),
        'mistakes_neg': print_vnames(var_name_fmt['mistake_neg'], n_points_neg),
        'fn': print_vnames(var_name_fmt['fn'], n_groups),
        'fp': print_vnames(var_name_fmt['fp'], n_groups),
        'fn_total': ['fn_total'],
        'fp_total': ['fp_total'],
        }

    if settings['parity_type'] in ('fpr', 'both'):
        names.update({
            'disc_fpr': [var_name_fmt['disc_fpr'] % pair for pair in mip_data['group_pairs']],
            'disc_total': ['disc_total']
            })

    if settings['parity_type'] in ('fnr', 'both'):
        names.update({
            'disc_fnr': [var_name_fmt['disc_fnr'] % pair for pair in mip_data['group_pairs']],
            'disc_total': ['disc_total']
            })


    add_variable(mip, name = names['theta_pos'], obj = 0.0, ub = theta_ub, lb = 0.0, vtype = 'C')
    add_variable(mip, name = names['theta_neg'], obj = 0.0, ub = 0.0, lb = theta_lb, vtype = 'C')
    add_variable(mip, name = names['theta_sign'], obj = 0.0, ub = 1.0, lb = 0.0, vtype = 'B')
    add_variable(mip, name = names['mistakes_pos'], obj = 0.0, ub = 1.0, lb = 0.0, vtype = 'B')
    add_variable(mip, name = names['mistakes_neg'], obj = 0.0, ub = 1.0, lb = 0.0, vtype = 'B')
    theta_names = names['theta_pos'] + names['theta_neg']

    # Cap L1 Norm for Coefficients
    # sum_j(theta_pos[j] - theta_neg[j]) = total_l1_norm
    cons.add(names = ['L1_norm_limit'],
             lin_expr = [SparsePair(ind = theta_names, val = [1.0] * n_variables + [-1.0] * n_variables)],
             senses = ['E'],
             rhs = [total_l1_norm])


    # Force Positive Coefficients or Negative Coefficients
    for T_pos, T_neg, s, tp, tn in zip(theta_ub, theta_lb, names['theta_sign'], names['theta_pos'], names['theta_neg']):
        # if tp[j] > 0 then s[j] = 1
        # if tn[j] < 0 then s[j] = 0
        # T_pos[j] * s[j] >= tp[j]       >>>>   T_pos[j] * s[j] - tp[j] >= 0
        # T_neg[j] * (1 - s[j]) >= -tn[j] >>>>  T_neg[j] * s[j] - tn[j] <= T_neg[j]
        cons.add(names = [f"set_{s}_pos", f"set_{s}_neg"],
                 lin_expr = [SparsePair(ind = [s, tp], val = [abs(T_pos), -1.0]),
                             SparsePair(ind = [s, tn], val = [abs(T_neg), -1.0])],
                 senses = ['G', 'L'],
                 rhs = [0.0, abs(T_neg)])

    # Define Mistake Indicators
    pos_vals = np.tile(mip_data['U_pos'], 2).tolist()
    for zp, val in zip(names['mistakes_pos'], pos_vals):

        # # if "z[i] = 0" IFF "score[i] >= margin_pos[i]" is active
        mip.indicator_constraints.add(name = f"def_{zp}_off",
                                      indvar = zp,
                                      complemented = 1,
                                      lin_expr = SparsePair(ind = theta_names, val = val),
                                      sense = 'G',
                                      rhs = abs(margin),
                                      indtype = 3)

        # if "z[i] = 1" -> "score[i] < margin_pos[i]" is active
        # mip.indicator_constraints.add(name = 'def_%r_on' % zp,
        #                               indvar = zp,
        #                               complemented = 0,
        #                               lin_expr = SparsePair(ind = theta_names, val = val),
        #                               sense = 'L',
        #                               rhs = abs(margin),
        #                               indtype = 1)


    neg_vals = np.tile(mip_data['U_neg'], 2).tolist()
    for zn, val in zip(names['mistakes_neg'], neg_vals):

        # if "z[i] = 0" IFF "score[i] <= margin_neg[i]" is active
        mip.indicator_constraints.add(name = f"def_{zn}_off",
                                      indvar = zn,
                                      complemented = 1,
                                      lin_expr = SparsePair(ind = theta_names, val = val),
                                      sense = 'L',
                                      rhs = -abs(margin),
                                      indtype = 3)

        # mip.indicator_constraints.add(name = 'def_%r_on' % zn,
        #                               indvar = zn,
        #                               complemented = 0,
        #                               lin_expr = SparsePair(ind = theta_names, val = val),
        #                               sense = 'G',
        #                               rhs = -abs(margin),
        #                               indtype = 1)

    # Old indicators
    # M_pos = margin + (total_l1_norm * np.max(abs(mip_data['U_pos']), axis = 1))
    # M_pos = M_pos.reshape((n_points_pos, 1))
    # assert np.all(M_pos > 0.0)
    #
    # M_neg = margin + (total_l1_norm * np.max(abs(mip_data['U_neg']), axis = 1))
    # M_neg = M_neg.reshape((n_points_neg, 1))
    # assert np.all(M_neg > 0.0)
    #
    # pos_vals = np.hstack((mip_data['U_pos'], mip_data['U_pos'], M_pos)).tolist()
    # for zp, val in zip(names['mistakes_pos'], pos_vals):
    #     cons.add(names = ['def_%r' % zp],
    #              lin_expr = [SparsePair(ind = theta_names + [zp], val = val)],
    #              senses = ['G'],
    #              rhs = margin)
    #
    # neg_vals = np.hstack((-mip_data['U_neg'], -mip_data['U_neg'], M_neg)).tolist()
    # for zn, val in zip(names['mistakes_neg'], neg_vals):
    #     cons.add(names = ['def_%r' % zn],
    #              lin_expr = [SparsePair(ind = theta_names + [zn], val = val)],
    #              senses = ['G'],
    #              rhs = [margin])
    #
    # set conflicting pairs
    # for zp, zn, i in zip(names['mistakes_pos'], names['mistakes_neg'], mip_data['conflicted_pairs']):
    #     cons.add(names = ['conflict_%d_%d' % (i[0], i[1])], lin_expr = [SparsePair(ind = [zp, zn], val = [1.0, 1.0])], senses = ['E'], rhs = [1.0])

    # define group FN
    for g in mip_data['group_labels']:
        fn_g = names['fn'][g]
        N_pos_g = mip_data['n_group_counts_pos'][g]
        I_pos_g = mip_data['group_indices']['pos'][g]
        z_pos_g = [names['mistakes_pos'][i] for i in I_pos_g]
        n_pos_g = mip_data['n_counts_pos'][I_pos_g]
        add_variable(cpx = mip, name = fn_g, obj = 0, ub = N_pos_g, lb = 0, vtype = 'I')
        cons.add(names = [f"def_{fn_g}"],
                 lin_expr = [SparsePair(ind = [fn_g] + z_pos_g, val = [1.0] + (-n_pos_g).tolist())],
                 senses = ['E'],
                 rhs = [0])

    # define group FP
    for g in mip_data['group_labels']:
        fp_g = names['fp'][g]
        N_neg_g = mip_data['n_group_counts_neg'][g]
        I_neg_g = mip_data['group_indices']['neg'][g]
        z_neg_g = [names['mistakes_neg'][i] for i in I_neg_g]
        n_neg_g = mip_data['n_counts_neg'][I_neg_g]
        add_variable(cpx = mip, name = fp_g, obj = 0, ub = N_neg_g, lb = 0, vtype = 'I')
        cons.add(names = [f"def_{fp_g}"],
                 lin_expr = [SparsePair(ind = [fp_g] + z_neg_g, val = [1.0] + (-n_neg_g).tolist())],
                 senses = ['E'],
                 rhs = [0])

    # define total FN
    add_variable(cpx = mip, name = names['fn_total'], obj = 0, ub = max_fn_total, lb = 0, vtype = 'I')
    cons.add(names = ['def_fn_total'],
             lin_expr = [SparsePair(ind = names['fn_total'] + names['fn'], val = [1.0] + [-1.0] * n_groups)],
             senses = ['E'],
             rhs = [0])

    # define total FP
    add_variable(cpx = mip, name = names['fp_total'], obj = 0, ub = max_fp_total, lb = 0, vtype = 'I')
    cons.add(names = ['def_fp_total'],
             lin_expr = [SparsePair(ind = names['fp_total'] + names['fp'], val = [1.0] + [-1.0] * n_groups)],
             senses = ['E'],
             rhs = [0])

    # add discrimination metric
    add_variable(cpx = mip, name = names['disc_total'], obj = 1.0, ub = 1.0, lb = 0.0, vtype = 'C')

    for a, b in mip_data['group_pairs']:

        if settings['parity_type'] in ('fnr', 'both'):

            # add disc_fnr[a][b] := fnr[a] - fnr[b]
            add_variable(mip, name = f"disc_fnr_{a}_{b}", obj = 0.0, ub = 1.0, lb = -1.0, vtype = 'C')
            fn_a, fn_b = names['fn'][a], names['fn'][b]
            N_a = mip_data['n_group_counts_pos'][a].astype(float)
            N_b = mip_data['n_group_counts_pos'][b].astype(float)
            cons.add(names = [f"def_disc_fnr_{a}_{b}"],
                     lin_expr = [SparsePair(ind = [f"disc_fnr_{a}_{b}", fn_a, fn_b], val = [N_a * N_b, -N_b, N_a])],
                     senses = ['E'],
                     rhs = [0])

            # link to overall discrimination
            cons.add(names = [f"disc_total_fnr_{a}_{b}"],
                     lin_expr = [SparsePair(ind = names['disc_total'] + [f"disc_fnr_{a}_{b}"], val = [1, -1])],
                     senses = ['G'],
                     rhs = [0])

        if settings['parity_type'] in ('fpr', 'both'):

            # add disc_fprr[a][b] := fnr[a] - fnr[b]
            add_variable(mip, name = f"disc_fpr_{a}_{b}", obj = 0.0, ub = 1.0, lb = -1.0, vtype = 'C')

            # add defining constraint
            fp_a, fp_b = names['fp'][a], names['fp'][b]
            N_a = mip_data['n_group_counts_neg'][a].astype(float)
            N_b = mip_data['n_group_counts_neg'][b].astype(float)
            cons.add(names = [f"def_disc_fpr_{a}_{b}"],
                     lin_expr = [SparsePair(ind = [f"disc_fpr_{a}_{b}", fp_a, fp_b], val = [N_a * N_b, -N_b, N_a])],
                     senses = ['E'],
                     rhs = [0])

            # link to overall discrimination
            cons.add(names = [f'disc_total_fpr_{a}_{b}'],
                     lin_expr = [SparsePair(ind = names['disc_total'] + [f"disc_fpr_{a}_{b}"], val = [1, -1])],
                     senses = ['G'],
                     rhs = [0])

    # collect information to validate solution
    info = {
        # performance
        'parity_type': settings['parity_type'],
        #
        'fpr_baseline': settings['fpr_baseline'],
        'fnr_baseline': settings['fnr_baseline'],
        'fpr_slack': settings['fpr_slack'],
        'fnr_slack': settings['fnr_slack'],
        #
        'max_fp_total': max_fp_total,
        'max_fn_total': max_fn_total,
        #
        # internal parameters
        'total_l1_norm': total_l1_norm,
        'margin': margin,
        #
        # helper functions
        'settings': dict(settings),
        'names': names,
        'variable_idx': {name: idx for idx, name in enumerate(mip_data['variable_names'])},
        'coefficient_idx': vars.get_indices(theta_names),
        'theta_pos_idx': vars.get_indices(names['theta_pos']),
        'theta_neg_idx': vars.get_indices(names['theta_neg']),
        'upper_bounds': {k: np.array(vars.get_upper_bounds(n)) for k, n in names.items()},
        'lower_bounds': {k: np.array(vars.get_lower_bounds(n)) for k, n in names.items()},
        }

    return mip, info

class LDALinearClassifierFitter:

    SETTINGS = {
        'parity_type': 'both',
        'fnr_slack': 0.0,
        'fpr_slack': 0.0,
        'total_l1_norm': 1.0,
        'margin': 0.0001,
        }

    PARITY_TYPES = {'fnr', 'fpr', 'both'}

    VAR_NAME_FMT = {
        'theta_pos': 'theta_pos_%d',
        'theta_neg': 'theta_neg_%d',
        'theta_sign': 'theta_sign_%d',
        'mistake_pos': 'lp_%d',
        'mistake_neg': 'ln_%d',
        'fn': 'fn_%d',
        'fp': 'fp_%d',
        'disc_fnr': 'disc_fnr_%d_%d',
        'disc_fpr': 'disc_fpr_%d_%d',
        }

    def __init__(self, X, y, G, data, sample_type, fnr_baseline, fpr_baseline, print_flag = True, **kwargs):
        """
        :param data:
        :param label_field:
        :param kwargs:
        """

        # check inputs
        assert 0.0 <= fnr_baseline <= 1.0
        assert 0.0 <= fpr_baseline <= 1.0
        assert 0.0 <= kwargs['fnr_slack'] <= (1.0 - fnr_baseline)
        assert 0.0 <= kwargs['fpr_slack'] <= (1.0 - fpr_baseline)
        assert kwargs['parity_type'] in self.PARITY_TYPES

        # create MIP
        self.mip_data = to_mip_data(X, y, G, data)
        cpx, info = build_lda_mip(mip_data = self.mip_data, var_name_fmt = self.VAR_NAME_FMT, fnr_baseline = fnr_baseline, fpr_baseline = fpr_baseline, **kwargs)

        # set default parameters
        p = cpx.parameters
        # p.emphasis.numerical.set(1)
        # p.mip.tolerances.integrality.set(0.0)
        p.mip.tolerances.mipgap.set(0.0)
        # p.mip.tolerances.absmipgap.set(0.0)
        # p.mip.limits.repairtries.set(10)
        # p.preprocessing.coeffreduce.set(0)

        #p.mip.display.set(print_flag)
        p.simplex.display.set(print_flag)
        p.paramdisplay.set(print_flag)
        # if not print_flag:
        #     cpx.set_results_stream(None)
        #     cpx.set_log_stream(None)
        #     cpx.set_error_stream(None)
        #     cpx.set_warning_stream(None)

        # attach fields
        self.names = info['names']
        self.mip = cpx
        self.info = info
        self.parity_type = str(kwargs['parity_type'])
        self.sample_type = str(sample_type)
        self.fnr_baseline = float(fnr_baseline)
        self.fpr_baseline = float(fpr_baseline)
        self.fnr_slack = float(kwargs['fnr_slack'])
        self.fpr_slack = float(kwargs['fpr_slack'])

    def solve(self, time_limit = 60, node_limit = None, return_stats = False, return_incumbents = False):
        """
        solves MIP
        #
        :param time_limit:
        :param node_limit:
        :param return_stats:
        :param return_incumbents:
        :return:
        """
        attach_stats_callback = return_stats or return_incumbents
        if attach_stats_callback:
            self._add_stats_callback(store_solutions = return_incumbents)

        # update time limit
        if time_limit is not None:
            self.mip = set_mip_time_limit(self.mip, time_limit)

        if node_limit is not None:
            self.mip = set_mip_node_limit(self.mip, node_limit)

        # solve
        self.mip.solve()

        info = self.solution_info

        if attach_stats_callback:
            progress_info, progress_incumbents = self._stats_callback.get_stats()
            info.update({'progress_info': progress_info, 'progress_incumbents': progress_incumbents})

        return info

    def populate(self, time_limit = 60.0):
        """
        populates solution pool
        :param max_gap: set to 0.0 to find equivalent solutions
        :param time_limit:
        :return:
        """
        p = self.mip.parameters
        p.mip.pool.replace.set(1)  # 1 = replace solutions with worst objective
        p.timelimit.set(float(time_limit))
        self.mip.populate_solution_pool()
        return True

    @property
    def solution(self):
        """
        :return: handle to CPLEX solution
        """
        # todo add wrapper if solution does not exist
        return self.mip.solution

    @property
    def solution_info(self):
        """returns information associated with the current best solution for the mip"""
        return get_mip_stats(self.mip)

    @property
    def has_solution(self):
        return has_solution(self.mip)

    def check_solution(self):
        assert check_mip_solution(mip = self.mip, info = self.info, data = self.data)

    def coefficients(self):
        s = self.solution
        theta_pos = np.array(s.get_values(self.names['theta_pos']))
        theta_neg = np.array(s.get_values(self.names['theta_neg']))
        return theta_pos + theta_neg

    def get_classifier(self):
        coefs = self.coefficients()
        intercept_idx = np.array(self.mip_data['intercept_idx'])
        coefficient_idx = np.array(self.mip_data['coefficient_idx'])

        # if self._mip_settings['standardize_data']:
        #     mu = np.array(self._mip_data['X_shift']).flatten()
        #     sigma = np.array(self._mip_data['X_scale']).flatten()
        #     coefs = coefs / sigma
        #     total_shift = coefs.dot(mu)
        #     coefficients = coefs[coefficient_idx]
        #     if intercept_idx >= 0:
        #         intercept = coefs[intercept_idx] - total_shift
        #     else:
        #         intercept = -total_shift

        b = coefs[intercept_idx]
        w = coefs[coefficient_idx]
        model = ClassificationModel(
                predict_handle = lambda X: np.sign(X.dot(w) + b),
                proba_handle = lambda X: expit(X.dot(w) + b),
                model_info = {
                    'model_type': ClassificationModel.LINEAR_MODEL_TYPE,
                    'intercept': b,
                    'coefficients': w,
                    },
                training_info = {
                    'method_name': 'mip',
                    'parity_type': self.parity_type,
                    }
                )

        return model

    def __repr__(self):
        s = [
            f'<LDALinearClassifierFitter(parity_type={self.parity_type}, fnr_baseline={self.fnr_baseline}, fpr_baseline={self.fpr_baseline}, fnr_slack={self.fnr_slack}, fpr_slack={self.fpr_slack})>',
            ]

        # add MIP info
        info = self.solution_info
        s.extend([f'{k}: {info[k]}' for k in ('has_solution', 'status')])
        s.extend([f'{k}: {info[k]:.2%}' for k in ('gap',)])
        s.extend([f'{k}: {info[k]:.4f}' for k in ('objval', 'upperbound', 'lowerbound')])
        s = '\n'.join(s)
        return s

    #### initialization
    def pull_coefficients(self, solution):
        """returns intercept + coefficients from a solution vector"""
        solution = np.array(solution)
        theta_pos = solution[self.info['theta_pos_idx']]
        theta_neg = solution[self.info['theta_neg_idx']]
        return theta_pos + theta_neg

    def fix_coefficients(self, values):
        """
        fixes the coefficient values
        :param values:
        :return:
        """
        vars = self.mip.variables
        theta_pos = np.maximum(values, 0.0)
        theta_neg = np.minimum(values, 0.0)
        vars.set_upper_bounds(list(zip(self.names['theta_pos'], theta_pos)))
        vars.set_lower_bounds(list(zip(self.names['theta_neg'], theta_neg)))

    def add_initial_solution(self, solution, objval = None, effort_level = 2, name = None, check_initialization = False):
        """
        :param solution: solution values to provide to the mip
        :param objval: objective value achieved by the solution. If provided, used to check the solution is accepted
        :param effort_level: integer describing the effort with which CPLEX will try to repair the solution
                            must be one of the values of mip.MIP_starts.effort_level)
                            1 <-> check_feasibility
                            2 <-> solve_fixed
                            3 <-> solve_MIP
                            4 <-> repair
                            5 <-> no_check
        :param name: name of the solution
        :return: The new cpx object with the solution added.
        """
        self.mip = add_mip_start(self.mip, solution, effort_level, name)

        if check_initialization and objval is not None:
            current_flag = self.print_flag
            if current_flag:
                self.print_flag = False
            self.solve(time_limit = 1.0)
            cpx_objval = self.solution.get_objective_value()
            if current_flag:
                self.print_flag = True
            try:
                assert np.less_equal(cpx_objval, objval)
            except AssertionError:
                warnings.warn('initial solution did not improve current upperbound\nCPLEX objval: %1.2f\nexpected_objval')

    def add_initial_solution_from_coefficients(self, coefs, effort_level = 2):
        """
        converts coefficient vector of linear classifier to a partial solution for the MIP
        :param coefs:
        :param effort_level: integer describing the effort with which CPLEX will try to repair the solution
                            must be one of the values of mip.MIP_starts.effort_level)
                            1 <-> check_feasibility
                            2 <-> solve_fixed
                            3 <-> solve_MIP
                            4 <-> repair
                            5 <-> no_check
        :return:
        """
        coefs = np.array(coefs).flatten()
        assert len(coefs) == len(self.names['theta_pos'])
        assert np.isfinite(coefs).all()
        coef_norm = np.abs(coefs).sum()
        if not np.isclose(coef_norm, self.info['total_l1_norm']):
            coefs = self.info['total_l1_norm'] * coefs / coef_norm

        assert np.isclose(np.abs(coefs).sum(), self.info['total_l1_norm'])
        sol = np.maximum(coefs, 0.0).tolist() + np.minimum(coefs, 0.0).tolist()
        idx = self.names['theta_pos'] + self.names['theta_neg']
        self.mip.MIP_starts.add(SparsePair(val = sol, ind = idx), effort_level)

    def solution_pool(self):
        """
        returns solution pool object for initialization
        """
        pool = SolutionPool()
        if self.has_solution:
            cpx_pool = self.solution.pool
            solutions = [cpx_pool.get_values(k) for k in range(cpx_pool.get_num())]
            objvals = [cpx_pool.get_objective_value(k) for k in range(cpx_pool.get_num())]
            coefficients = list(map(self.pull_coefficients, solutions))
            pool.add(solution = solutions, coefficients = coefficients, objval = objvals)
        return pool

    #### generic MIP fields ####
    def _add_stats_callback(self, store_solutions = False):
        if not hasattr(self, '_stats_callback'):
            sol_idx = self.names['theta_pos'] + self.names['theta_neg']
            min_idx, max_idx = min(sol_idx), max(sol_idx)
            assert np.array_equal(np.array(sol_idx), np.arange(min_idx, max_idx + 1))
            cb = self.mip.register_callback(StatsCallback)
            cb.initialize(store_solutions, solution_start_idx = min_idx, solution_end_idx = max_idx)
            self._stats_callback = cb

    def set_parallelization(self, threads = 0):
        """
        toggles parallelization in CPLEX
        :param threads: number of threads to use; set to 0 to choose automatically
        :param cpx: Cplex object
        :param flag: True to turn off MIP display
        :return:
        """
        assert isinstance(threads, int) and threads >= 0
        p = self.mip.parameters
        if threads == 0:
            p.parallel.set(0)
            p.threads.set(0)
        else:
            p.parallel.set(1) #-1 = opportunistic\n  0 = automatic\n  1 = deterministic'
            p.threads.set(threads)



    # def enumerate_equivalent_solutions(self, pool, time_limit = 30):
    #     """
    #     :param pool:
    #     :param time_limit:
    #     :return:
    #     """
    #
    #     # get mistake list
    #     mistake_names = self.names['mistakes_pos'] + self.names['mistakes_neg']
    #     z = self.solution.get_values(mistake_names)
    #
    #     info = self.solution_info
    #     lb = info['lowerbound']
    #     objval = info['objval']
    #
    #     # remove original solution from the equivalent models finder
    #     eq_mip = ZeroOneLossMIP(self.original_data, print_flag = self.print_flag, parallel_flag = self.parallel_flag, random_seed = self.random_seed)
    #     eq_mip.add_mistake_cut(z)
    #     eq_mip.set_total_mistakes(lb = lb)
    #
    #     # search for equivalent models
    #     equivalent_output = []
    #     keep_looking = True
    #     while keep_looking:
    #         out = eq_mip.solve(time_limit = time_limit)
    #         if out['objval'] < objval:
    #             warnings.warn('found new solution with better objval than baseline')
    #             equivalent_output.append(out)
    #             eq_mip.add_mistake_cut()
    #         elif out['objval'] == objval:
    #             equivalent_output.append(out)
    #             eq_mip.add_mistake_cut()
    #         else:
    #             keep_looking = False
    #         pool.add_from_mip(eq_mip)
    #
    #     return equivalent_output, pool
    #

