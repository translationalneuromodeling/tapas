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

import sklearn
import sklearn.preprocessing, sklearn.decomposition, sklearn.manifold

import shap

import matplotlib as mpl, matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='ticks', palette='muted', color_codes=True)

import genbed.utilities as utils

###############################################################################

def visualise_data(
        data, labels=None, *, confounds=None, demean_confounds=True):
    """
    Plots several summaries of a data set.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Shape [n_observations, n_features].
    labels : pandas.Series, optional
        Shape [???,]. Must be indexable by `data.index`.
    
    Other Parameters
    ----------------
    confounds : pandas.DataFrame, optional
        Shape [???, n_confounds]. Must be indexable by `data.index`.
    demean_confounds : bool, optional
        If `True`, confounds are normalised along the features axis.
    """
    # samples = Panel [n_samples, n_observations, n_features]
    
    data, labels, confounds = utils.sanitise_inputs(data, labels, confounds)
    n_o, n_f = data.shape
    if labels is not None:
        n_c = len(labels.cat.categories)
    else:
        n_c = 1  # All same class
    if confounds is not None:
        data = utils.remove_confounds(data, confounds, demean_confounds)
    
    #--------------------------------------------------------------------------
    
    # Plot data distributions
    plot_kws = {}
    if n_o > 100:
        plot_kws['s'] = 5; plot_kws['edgecolor'] = None
    if n_f <= 10:
        # Full interactions for small data
        if labels is None:
            fig = sns.pairplot(data, plot_kws=plot_kws)
        else:
            fig = sns.pairplot(
                    data.join(labels), hue=labels.name,
                    vars=data.columns, plot_kws=plot_kws)
            # Need to specify vars as pairplot checks values not
            # categorical-ness (e.g. labels=[1,1,2,3] is numeric...)
            # https://github.com/mwaskom/seaborn/issues/919#issuecomment-366872386
    elif n_f * n_c <= 75:
        # For intermediate data, just plot marginals
        norm_data = data.apply(sklearn.preprocessing.scale)
        if labels is None:
            fig = sns.catplot(
                    data=norm_data,
                    kind='violin', inner='quartile')
        else:
            plot_data = norm_data.join(labels).melt(
                    labels.name, var_name='Feature', value_name='Value')
            fig = sns.catplot(
                    data=plot_data, x='Feature', y='Value', hue=labels.name,
                    order=data.columns, kind='boxen', legend_out=False)
        fig.set_xticklabels(rotation=45, horizontalalignment='right')
        fig.ax.set_title("Feature distributions")
    else:
        # Top PCA components for big data
        norm_data = data.apply(sklearn.preprocessing.scale)
        pca = sklearn.decomposition.PCA(n_components=5).fit_transform(norm_data)
        pca = pd.DataFrame(
                data=pca, index=data.index,
                columns=['PCA: #{:d}'.format(i) for i in range(pca.shape[1])])
        if labels is None:
            fig = sns.pairplot(pca, plot_kws=plot_kws)
        else:
            fig = sns.pairplot(
                    pca.join(labels), hue=labels.name,
                    vars=pca.columns, plot_kws=plot_kws)
    
    try:
        fig.fig.tight_layout()
        # This makes sure we don't lose any axes labels etc, but also makes the
        # legend almost impossible to see for `sns.pairplot()`... However,
        # there is currently no way to customise the legend creation so it
        # stays for now.
    except NameError:
        pass
    
    #--------------------------------------------------------------------------
    
    # Correlations between features
    if n_f <= 200:
        fig, ax = plt.subplots()
        ax = sns.heatmap(
                data.corr(), ax=ax, square=True,
                vmin=-1.0, center=0.0, vmax=1.0,
                cmap='RdBu_r', cbar_kws={'label': "Correlation coefficient"})
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title("Feature correlations")
        fig.tight_layout()
    
    #--------------------------------------------------------------------------
    
    return

###############################################################################

def visualise_manifold(
        data, labels=None, *, confounds=None, demean_confounds=True,
        normalise=True, manifold=sklearn.manifold.TSNE(init='pca')):
    """
    Plots the data set on a low-dimensional manifold.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Shape [n_observations, n_features].
    labels : pandas.Series, optional
        Shape [???,]. Must be indexable by `data.index`.
    
    Other Parameters
    ----------------
    confounds : pandas.DataFrame, optional
        Shape [???, n_confounds]. Must be indexable by `data.index`.
    demean_confounds : bool, optional
        If `True`, confounds are normalised along the features axis.
    normalise : bool, optional
        If `True`, data is normalised along the features axis.
    manifold : optional
        Instance of a class from `sklearn.manifold` or `sklearn.decomposition`.
        Use this to pass in a different algorithm for dimensionality reduction.
    """
    # samples = Panel [n_samples, n_observations, n_features]
    
    data, labels, confounds = utils.sanitise_inputs(data, labels, confounds)
    n_o, n_f = data.shape
    if labels is not None:
        n_c = len(labels.cat.categories)
    else:
        n_c = 1  # All same class
    if confounds is not None:
        data = utils.remove_confounds(data, confounds, demean_confounds)
    
    # Preprocess data
    if normalise:
        data = data.apply(sklearn.preprocessing.scale)
    # Concatenate samples...
    if n_f > 50:
        # Reduce data before passing to manifold?
        pass
    
    # Find the data in the embedding space
    manifold.set_params(n_components=2)  # Sanity check...
    embedding = manifold.fit_transform(data)
    # Separate samples...
    
    # And plot
    fig, ax = plt.subplots()
    # Plot samples...
    if labels is None:
        ax.plot(embedding[:, 0], embedding[:, 1], 'o', markersize=10)
    else:
        for c in range(n_c):
            ax.plot(
                    embedding[labels.cat.codes == c, 0],
                    embedding[labels.cat.codes == c, 1],
                    'o', markersize=10, markeredgecolor='k',
                    label=labels.cat.categories[c])
        ax.legend(title=labels.name)
    
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Data manifold")
    fig.tight_layout()
    
    return

###############################################################################

def visualise_performance(labels, predictions, probabilities=None):
    """
    Plots several summary metrics of classification performance.
    
    Parameters
    ----------
    labels : pandas.Series
        Shape [???,]. Must be indexable by `predictions.index.levels[-1]`.
    predictions : pandas.Series
        Shape [[n_folds, fold_size],] (i.e. a MultiIndex over folds).
    probabilities : pandas.DataFrame, optional
        Shape [[n_folds, fold_size], n_classes].
    """
    # samples = Panel [n_samples, n_observations, n_features]
    
    predictions = utils.to_categorical(predictions)
    labels = labels[predictions.index.levels[-1]]
    labels = utils.to_categorical(labels)
    unique_labels = labels.sort_values()
    labels = labels.reindex(predictions.index, level=-1)
    
    #--------------------------------------------------------------------------
    
    def plot_confusion_matrix(labels, predictions, normalise):
        if normalise:
            C = pd.crosstab(labels, predictions, normalize='index')
            vmax = 1.0; fmt = '.2f'; clabel = "Proportion"
        else:
            C = pd.crosstab(labels, predictions)
            vmax = C.values.max(); fmt = 'd'; clabel = "Count"
        
        fig, ax = plt.subplots()
        ax = sns.heatmap(
                C, ax=ax, square=True, annot=True, fmt=fmt,
                vmin=0.0, vmax=vmax, linewidths=0.5,
                cmap='Blues', cbar_kws={'label': clabel})
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        if normalise:
            ax.set_title("Normalised confusion matrix")
        else:
            ax.set_title("Confusion matrix")
        fig.tight_layout()
        
        return fig, ax
    
    plot_confusion_matrix(labels, predictions, normalise=True)
    plot_confusion_matrix(labels, predictions, normalise=False)
    
    #--------------------------------------------------------------------------
    
    # Plot per-observation probabilities
    if probabilities is not None:
        n_c = len(unique_labels.cat.categories)
        n_x = len(unique_labels)  # x_locs = list(range(n_x))
        
        fig, ax = plt.subplots()
        
        # Plot class boundaries and labels
        changes = np.where(np.diff(unique_labels.cat.codes))[0] + 0.5
        for change in changes:
            ax.plot([change,]*2, [0.0, 1.0], 'k')
        changes = np.concatenate([[-0.5,], changes, [n_x - 0.5,]])
        ticks = changes[:-1] + np.diff(changes) / 2.0
        ax.set_xticks(ticks); ax.set_xticklabels(unique_labels.cat.categories)
        # N.B. Categorical sorting happens on `codes` not `categories`
        # (i.e. classes still appear in this order, even after `sort_values()`)
        
        # Null line
        ax.plot([-1.0, n_x], [1.0 / n_c,]*2, color=[0.7,]*3)
        
        # Plot the data - repeated predictions are plotted at the same `x_pos`
        # https://stackoverflow.com/a/8251668
        label_inds = np.argsort(unique_labels.index)
        pred_inds = np.searchsorted(
                unique_labels.index[label_inds],
                predictions.index.get_level_values(-1))
        x_pos = label_inds[pred_inds]
        for cat in unique_labels.cat.categories:
            ax.plot(x_pos, probabilities[cat], 'o', label=cat,
                    markeredgecolor='k', markeredgewidth=0.5)
        ax.legend(title="Prediction")
        
        ax.set_xlim(-1.0, n_x)
        # Crop redundant part of plot for two-class case
        if n_c == 2:
            ax.set_ylim(0.5, 1.0 + 0.025)
        else:
            ax.set_ylim(-0.025, 1.0 + 0.025)
        ax.set_xlabel("True class")
        ax.set_ylabel("Predicted class probability")
        ax.set_title("Class probabilities")
        fig.tight_layout()
    
    #--------------------------------------------------------------------------
    
    return

###############################################################################

def visualise_null_metrics(metrics, *, name='Metric', **kwargs):
    """
    Visualises a null distribution of classification performance.
    
    Parameters
    ----------
    metrics : np.array
        Length [n_perms+1,]. Each item is a metric of classification
        performance, with `metric[0]` being the unshuffled result (typically
        the output of `classification.evaluate_significance()`).
    name : str, optional
        The name of the metric to be plotted (e.g. 'Balanced accuracy').
    **kwargs : optional
        Passed to `seaborn.distplot()`.
    """
    
    fig, ax = plt.subplots()
    sns.histplot(
            metrics[1:], ax=ax,
            stat='density', kde=True, kde_kws={'cut': 3},
            label='Null', **kwargs)
    ax.plot([metrics[0],]*2, ax.get_ylim(), 'r', linewidth=3.0, label='True')
    ax.legend()
    ax.set_xlabel(name); ax.set_ylabel("Probability density")
    ax.set_title('Classification null distribution')
    fig.tight_layout()
    
    return

###############################################################################

def visualise_explanation(explanation, per_class=True):
    """
    Visualises an explanation of classification performance.
    
    Parameters
    ----------
    explanation
        Output of `classification.explain_classifier()`.
    per_class : bool, optional
        Whether to also plot explanations at the level of the individual
        classes, or just the summary.
    """
    
    # Summarise across all classes
    fig, ax = plt.subplots()
    shap.summary_plot(
            explanation.shap_values,
            explanation.data,
            class_names=explanation.clf_categories,
            max_display=15)
    ax.set_ylabel("Feature")
    ax.set_title("Classifier feature weightings (all classes)")
    fig.tight_layout()
    
    # And then break down by class
    if per_class:
        for i in range(len(explanation.shap_values)):
            # Summary plot: break down by feature
            fig, ax = plt.subplots()
            shap.summary_plot(
                    explanation.shap_values[i],
                    explanation.data,
                    max_display=10)
            ax.set_ylabel("Feature")
            ax.set_title(
                    "Classifier feature weightings (class: {})"
                    .format(explanation.clf_categories[i]))
            fig.tight_layout()
            
            # Decision plot: break down by observation
            fig, ax = plt.subplots()
            shap.decision_plot(
                    explanation.explainer.expected_value[i],
                    explanation.shap_values[i],
                    link='logit', features=explanation.data)
            ax.set_ylabel("Feature")
            ax.set_title(
                    "Classifier decision weightings (class: {})"
                    .format(explanation.clf_categories[i]))
            fig.tight_layout()
    
    return

###############################################################################
