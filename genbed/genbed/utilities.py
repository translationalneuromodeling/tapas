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

###############################################################################

import warnings

import numpy as np
import pandas as pd

import sklearn
import sklearn.preprocessing, sklearn.linear_model

###############################################################################
# Input checks / conversions

def to_categorical(labels, to_string=False):
    """
    Converts to categorical and does some sanity checks.
    
    Parameters
    ----------
    labels : pandas.Series
    to_string : bool, optional
        Whether to convert the categories to strings.
    
    Returns
    -------
    labels : pandas.Series
        Input, but converted to categorical.
    """
    if not isinstance(labels, pd.Series):
        raise TypeError("`labels` should be a `pandas.Series`")
    
    if not hasattr(labels, 'cat'):
        labels = labels.astype('category')
    else:
        labels = labels.cat.remove_unused_categories()
    
    # Convert categories to string representations
    if to_string:
        labels.cat.categories = labels.cat.categories.astype(str)
    
    return labels

#------------------------------------------------------------------------------

def sanitise_inputs(data, labels=None, confounds=None):
    """
    Checks, reorders and cleans key inputs.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Shape [n_observations, n_features].
    labels : pandas.Series, optional
        Shape [???,]. Must be indexable by `data.index`.
    confounds : pandas.DataFrame, optional
        Shape [???, n_confounds]. Must be indexable by `data.index`.
    
    Returns
    -------
    (data, labels, confounds)
        These will have a consistent index order, and will be converted to
        float / categorical dtypes as appropriate.
    """
    # samples = Panel [n_samples, n_observations, n_features]
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` should be a `pandas.DataFrame`")
    data = data.astype(np.float64)
    if data.isna().any(axis=None):
        warnings.warn("sanitise_inputs(): Removing NaN values from `data`.", RuntimeWarning)
        data = data.fillna(data.mean())
    
    if labels is not None:
        if not isinstance(labels, pd.Series):
            raise TypeError("`labels` should be a `pandas.Series`")
        diffs = set(labels.index).difference(data.index)
        if len(diffs) > 0:
            warnings.warn("sanitise_inputs(): Removing {:d} observations from `labels` not present in `data`.".format(len(diffs)), RuntimeWarning)
        labels = labels.loc[data.index]  # Need explicit `.loc` to error on missing
        labels = to_categorical(labels)
        if labels.name is None:
            labels.name = 'Class'
    
    if confounds is not None:
        if not isinstance(confounds, pd.DataFrame):
            raise TypeError("`confounds` should be a `pandas.DataFrame`")
        diffs = set(confounds.index).difference(data.index)
        if len(diffs) > 0:
            warnings.warn("sanitise_inputs(): Removing {:d} observations from `confounds` not present in `data`.".format(len(diffs)), RuntimeWarning)
        confounds = confounds.loc[data.index, :]
        confounds = confounds.astype(np.float64)
        if confounds.isna().any(axis=None):
            warnings.warn("sanitise_inputs(): Removing NaN values from `confounds`.", RuntimeWarning)
            confounds = confounds.fillna(confounds.mean())
    
    return data, labels, confounds

###############################################################################

def remove_confounds(data, confounds, demean=True):
    """
    Linearly regresses `confounds` from `data`.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Shape [n_observations, n_features].
    confounds : pandas.DataFrame
        Shape [???, n_confounds]. Must be indexable by `data.index`.
    demean : bool, optional
        Whether to remove the means from `confounds` before the regression.
    
    Returns
    -------
    cleaned_data : pandas.DataFrame
        Shape [n_observations, n_features].
    """
    # samples = Panel [n_samples, n_observations, n_features]
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` should be a `pandas.DataFrame`")
    if not isinstance(confounds, pd.DataFrame):
        raise TypeError("`confounds` should be a `pandas.DataFrame`")
    
    # Preprocess confounds
    confounds = confounds.loc[data.index, :]
    confounds = confounds.apply(
            lambda x: sklearn.preprocessing.scale(x, with_mean=demean))
    # `with_mean`: If True, center the data before scaling.
    
    # And regress out of data
    # `fit_intercept=False` means we don't remove the column means from the
    # data (unless `demean=False` and the mean is a confound)
    regression = sklearn.linear_model.LinearRegression(fit_intercept=False)
    regression.fit(confounds, data)
    cleaned_data = data - regression.predict(confounds)
    
    return cleaned_data

###############################################################################
