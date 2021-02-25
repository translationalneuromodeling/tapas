# -*- coding: utf-8 -*-

# genbed: A Python toolbox for generative embedding based classification

# Copyright 2019 Sam Harrison
# Translational Neuromodeling Unit, University of Zurich & ETH Zurich
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

###############################################################################

import numpy as np
import pandas as pd
import scipy, scipy.stats
import statsmodels
import statsmodels.stats.multitest as mt

import sklearn
import sklearn.preprocessing

import genbed.utilities as utils

###############################################################################

def run_test(data, test, *args, groups=None, **kwargs):
    """
    Runs test on each column of data, correcting for multiple comparisons.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Shape [n_observations, n_features].
    test : function
        Called as `test(data, *args, **kwargs)`, and must return the test
        statistic and p-value (as per the `scipy.stats` API).
    groups : pd.Series, optional
        If this is present, `data` is stratified based on group, and the
        call is then `test(data[g1], data[g2], ...,  *args, **kwargs)`.
    
    Returns
    -------
    statistics : pandas.Series
        Shape [n_features,].
    pvalues : pandas.Series
        Shape [n_features,].
    """
    
    data, groups, _ = utils.sanitise_inputs(data, groups, None)
    
    statistics = np.full([data.shape[1]], np.NaN)
    pvalues    = np.full([data.shape[1]], np.NaN)
    for c,col in enumerate(data.columns):
        if groups is None:
            [statistics[c], pvalues[c]] = test(data[col], *args, **kwargs)
        else:
            [statistics[c], pvalues[c]] = test(
                    *[data.loc[groups==g, col] for g in groups.cat.categories],
                    *args, **kwargs)
    _, pvalues, _, _ = mt.multipletests(pvalues)
    
    statistics = pd.Series(data=statistics, index=data.columns)
    pvalues    = pd.Series(data=pvalues, index=data.columns)
    
    return statistics, pvalues

###############################################################################

def print_test_results(statistics, pvalues, *, alpha=0.05, indent=4):
    """
    Prints the results of a set of univariate tests (see `run_test()`).
    
    Parameters
    ----------
    statistics : pandas.Series
        Shape [n_features,].
    pvalues : pandas.Series
        Shape [n_features,].
    alpha : float, optional
        Only print results of tests where `p < alpha`.
    
    Other Parameters
    ----------------
    indent : int, optional
        Number of spaces to indent results.
    """
    
    if not isinstance(statistics, pd.Series):
        raise TypeError("`statistics` should be a `pandas.Series`")
    if not isinstance(pvalues, pd.Series):
        raise TypeError("`pvalues` should be a `pandas.Series`")
    
    prefix = " " * indent
    
    # Return early if no significant tests
    if min(pvalues) > alpha:
        print(prefix + "No tests significant at alpha={:.2f}".format(alpha))
        return
    
    # Otherwise print p < alpha
    pad = max([len(name) for name in statistics.index])
    for name in statistics.index:
        if pvalues[name] <= alpha:
            print(prefix + "{:>{pad}}: {:+.2f} [p={:.2f}]"
                  .format(name, statistics[name], pvalues[name], pad=pad))
    
    return

###############################################################################

def run_tests(
        data, labels=None,
        *, confounds=None, demean_confounds=True, alpha=0.05):
    """
    Runs several simple statistical tests, primarily for group differences.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Shape [n_observations, n_features].
    labels : pandas.Series
        Shape [???,]. Must be indexable by `data.index`.
    
    Other Parameters
    ----------------
    confounds : pandas.DataFrame, optional
        Shape [???, n_confounds]. Must be indexable by `data.index`.
    demean_confounds : bool, optional
        If `True`, confounds are normalised along the features axis.
    alpha : float, optional
        Only print results of tests where `p < alpha`.
    """
    
    # Dummy labels (i.e. just a single group)
    if labels is None:
        labels = ['Group'] * data.shape[0]
        labels = pd.Series(data=labels, index=data.index, dtype='category')
    
    data, labels, confounds = utils.sanitise_inputs(data, labels, confounds)
    if confounds is not None:
        data = utils.remove_confounds(data, confounds, demean_confounds)
    
    print("Running statistical tests...")
    print("Correction for multiple comparisons is across features, not tests.")
    print("No. of features: {:d}".format(data.shape[1]))
    print("No. of observations: {:d}".format(data.shape[0]))
    if confounds is not None:
        print("No. of confounds: {:d}".format(confounds.shape[1]))
    print("No. of classes: {:d}".format(len(labels.cat.categories)))
    print("Classes: {}".format(", ".join(map(str, labels.cat.categories))))
    print()
    
    # Test groups individually
    for cat in labels.cat.categories:
        print("{}: {}".format(labels.name, cat))
        c_data = data.loc[labels == cat, :]
        print("No. of observations: {:d}".format(c_data.shape[0]))
        
        # Are observations normally distributed?
        # https://doi.org/10.1080/00949655.2010.520163
        print("Shapiro-Wilk test for non-normality:")
        s, p = run_test(c_data, scipy.stats.shapiro)
        print_test_results(s, p, alpha=alpha)
        
        print()
    
    # And test for group differences etc...
    if len(labels.cat.categories) >= 2:
        #print("ANOVA for difference in group means:")
        #s, p = run_test(data, scipy.stats.f_oneway, groups=labels)
        #print_test_results(s, p, alpha=alpha)
        #print()
        
        print("Kruskal-Wallis H-test for difference in group medians:")
        s, p = run_test(data, scipy.stats.kruskal, groups=labels)
        print_test_results(s, p, alpha=alpha)
        print()
        
        print("Levene test for difference in group variances:")
        s, p = run_test(data, scipy.stats.levene, groups=labels)
        print_test_results(s, p, alpha=alpha)
        print()
    
    else:
        # Not clear from the docs if `scipy.stats.wilcoxon` is a suitable
        # alternative in the one-sample case
        print("t-test for non-zero means:")
        s, p = run_test(data, scipy.stats.ttest_1samp, popmean=0.0)
        print_test_results(s, p, alpha=alpha)
        print()
    
    return

###############################################################################
