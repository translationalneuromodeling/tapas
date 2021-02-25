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

import builtins
import collections
import types
import warnings

import numpy as np
import pandas as pd
import scipy, scipy.stats

import sklearn
import sklearn.preprocessing, sklearn.linear_model
import sklearn.feature_selection, sklearn.model_selection
import sklearn.pipeline, sklearn.metrics

import shap

import genbed.utilities as utils

# https://stackoverflow.com/a/26433913
def _warning_format(message, category, filename, lineno, line=None):
    return "{}:{}: {}:\n  {}\n".format(filename, lineno, category.__name__, message)
warnings.formatwarning = _warning_format

warnings.filterwarnings('ignore', 'Solver terminated early.*')
warnings.filterwarnings('ignore', 'The max_iter was reached*')

###############################################################################

defaults = types.SimpleNamespace()

defaults.feature_reduction = sklearn.decomposition.PCA(
        n_components='mle', whiten=True)  # mle: Maximum likelihood estimate
defaults.feature_reduction_params = {}

defaults.feature_selection = sklearn.feature_selection.SelectPercentile()
defaults.feature_selection_params = {
        'percentile': np.geomspace(1.0, 100.0, 10)}

defaults.classifier = sklearn.linear_model.LogisticRegression(
            multi_class='multinomial', class_weight='balanced',
            penalty='elasticnet', l1_ratio=0.0,
            solver='saga')
defaults.classifier_params = {
        'C': np.geomspace(1.0e-2, 1.0e2, 5)}
#defaults.classifier_params = {
#        'C': np.geomspace(1.0e-2, 1.0e2, 9),
#        'l1_ratio': np.linspace(0.0, 1.0, 6)}

#------------------------------------------------------------------------------

def get_default_classifier(
        classifier=None, parameters=None,
        *, feature_reduction=False, feature_selection=False):
    """
    Builds the default classifier, along with hyperparameters to optimise.
    
    Parameters
    ----------
    classifier : optional
        A classifier (or pipeline) conforming to the sklearn interface. This
        allows a custom classifier to be combined with the feature reduction
        and selection steps. If not specified, `defaults.classifier` is used.
    parameters : dict, optional
        A set of hyperparameters to tune for the classifier. These should
        conform to the sklearn `GridSearchCV` interface. If neither
        `classifier` or `parameters` is specified, `defaults.classifier_params`
        is used.
    feature_reduction: bool
        Whether to add `defaults.feature_reduction` to the classification
        pipeline.
    feature_selection: bool
        Whether to add `defaults.feature_selection` to the classification
        pipeline.
    
    Returns
    -------
    classifier : sklearn.pipeline.Pipeline
    parameters : dict
    """
    
    pipeline_steps = []; pipeline_parameters = {}
    def update_pipeline(name, step, params={}):
        pipeline_steps.append((name, step))
        pipeline_parameters.update(
                {name+'__'+key: value for key, value in params.items()})
        return
    
    if feature_reduction:
        fr = defaults.feature_reduction
        fr_params = defaults.feature_reduction_params
        update_pipeline('fr', fr, fr_params)
    
    if feature_selection:
        fs = defaults.feature_selection
        fs_params = defaults.feature_selection_params
        update_pipeline('fs', fs, fs_params)
    
    if classifier is None:
        # Use defaults if necessary
        clf = defaults.classifier
        if parameters is None:
            clf_params = defaults.classifier_params
        else:
            clf_params = parameters
        update_pipeline('clf', clf, clf_params)
    elif isinstance(classifier, sklearn.pipeline.Pipeline):
        # Concatenate pipelines if necessary
        parameters = parameters if parameters is not None else {}
        for name, step in classifier.named_steps.items():
            update_pipeline(name, step)
        pipeline_parameters.update(parameters)
    else:
        # Otherwise, just the final classifier has been provided
        clf = classifier
        clf_params = parameters if parameters is not None else {}
        update_pipeline('clf', clf, clf_params)
    
    # Put the pipeline together
    pipeline = sklearn.pipeline.Pipeline(pipeline_steps)
    
    return pipeline, pipeline_parameters

###############################################################################

def predict(
        data, labels, classifier, parameters={},
        *, confounds=None, demean_confounds=True, normalise=True,
        cv_iter=None, groups=None, inner_cv_kwargs={},
        return_probabilities=False, verbose=True):
    """
    Predicts `labels` from `data` using nested cross-validation.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Shape [n_observations, n_features].
    labels : pandas.Series
        Shape [???,]. Must be indexable by `data.index`.
    classifier
        A classifier (or pipeline) conforming to the sklearn interface. See
        `get_default_classifier()`.
    parameters : dict, optional
        A set of hyperparameters to tune, conforming to the sklearn
        `GridSearchCV` interface.
    
    Returns
    -------
    predictions : pandas.Series
        Shape [[n_folds, fold_size],] (i.e. a MultiIndex over folds).
    probabilities : pandas.DataFrame, optional
        Shape [[n_folds, fold_size], n_classes]. Only returned if
        `return_probabilities` is set.
    
    Other Parameters
    ----------------
    confounds : pandas.DataFrame, optional
        Shape [???, n_confounds]. Must be indexable by `data.index`.
    demean_confounds : bool, optional
        If `True`, confounds are normalised along the features axis (respecting
        the train/test split). This is almost always needed to stop e.g.
        conflation of mean effects and the confound variance.
    normalise : bool, optional
        If `True`, data is normalised along the features axis (respecting the
        train/test split).
    cv_iter : optional
        A cross-validation generator from `sklearn.model_selection`.
        Default: `StratifiedKFold`.
    groups : pandas.Series, optional
        Shape [???,]. Must be indexable by `data.index`. Passed to `cv_iter` to
        allow for stratification based on group membership.
    inner_cv_kwargs : dict, optional
        Passed to `optimise_classifier()` to modify the inner
        cross-validation loop for hyperparameter tuning.
    return_probabilities : bool, optional
        Whether to calculate the class probabilities from the classifier (which
        must therefore implement `predict_proba()`).
    verbose : bool, optional
    """
    # samples = Panel [n_samples, n_observations, n_features]
    print = builtins.print if verbose else lambda *a, **k: None
    
    data, labels, confounds = utils.sanitise_inputs(data, labels, confounds)
    # Working with codes is safer (e.g. sklearn doesn't like `dtype=object`)
    codes = labels.cat.codes
    if groups is not None:
        groups = groups[data.index]
        groups = groups.astype('category').cat.codes
    
    print("Classifying data...")
    print("No. of features: {:d}".format(data.shape[1]))
    print("No. of observations: {:d}".format(data.shape[0]))
    if confounds is not None:
        print("No. of confounds: {:d}".format(confounds.shape[1]))
    print("No. of classes: {:d}".format(len(labels.cat.categories)))
    print("Classes: {}".format(", ".join(map(str, labels.cat.categories))))
    print()
    
    # Initialise storage for results
    predictions = []
    if return_probabilities:
        probabilities = []
    
    if cv_iter is None:
        # Try to keep at least one class in each fold
        n_splits=min(labels.value_counts())
        n_splits=min(10, max(3, n_splits))
        cv_iter = sklearn.model_selection.StratifiedKFold(
                n_splits=n_splits, shuffle=True)
    
    # Outer cross-validation loop
    n = 1; n_folds = cv_iter.get_n_splits(data, codes, groups)
    for train_inds, test_inds in cv_iter.split(data, codes, groups):
        # Train / test split
        train_inds, test_inds = data.index[train_inds], data.index[test_inds]
        X_train, X_test = data.loc[train_inds, :], data.loc[test_inds, :]
        y_train, y_test = codes[train_inds], codes[test_inds]
        
        if confounds is not None:
            C_train = confounds.loc[train_inds, :]
            C_test  = confounds.loc[test_inds, :]
            # Normalise (respecting decision to remove means)
            scaler  = sklearn.preprocessing.StandardScaler(
                    with_mean=demean_confounds)
            scaler.fit(C_train)
            # `with_mean`: If True, center the data before scaling.
            C_train = scaler.transform(C_train)
            C_test  = scaler.transform(C_test)
            # Remove confounds (but don't remove the data means)
            regression = sklearn.linear_model.LinearRegression(fit_intercept=False)
            regression.fit(C_train, X_train)
            X_train = X_train - regression.predict(C_train)
            X_test  = X_test  - regression.predict(C_test)
        
        if normalise:
            scaler  = sklearn.preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test  = scaler.transform(X_test)
        
        # Fit to data!
        clf = optimise_classifier(
                X_train, y_train, classifier, parameters, **inner_cv_kwargs)
        #clf.fit(X_train, y_train)
        
        # Record performance
        pred = pd.Categorical.from_codes(
                clf.predict(X_test), labels.cat.categories)
        predictions.append(
                pd.Series(
                    data=pred, index=test_inds,
                    name='Predicted_'+labels.name))
        if return_probabilities:
            clf_categories = labels.cat.categories[clf.best_estimator_.classes_]
            probabilities.append(
                    pd.DataFrame(
                        data=clf.predict_proba(X_test),
                        index=test_inds, columns=clf_categories))
        
        print("Finished iteration {} of {}".format(n, n_folds))
        if len(clf.best_params_) > 0:
            print(clf.best_params_)
        n += 1
    
    # Turn into multiindex over folds
    predictions = pd.concat(
            predictions, axis='index', keys=range(n_folds),
            names=['Fold', data.index.name])
    if return_probabilities:
        probabilities = pd.concat(
                probabilities, axis='index', keys=range(n_folds),
                names=['Fold', data.index.name])
        # If not all classes in a given `y_train`, then the missing classes get
        # filled as NaN when concatenating. This is probably what we want: it's
        # a clear flag that the classifier is solving a different problem
        #probabilities = probabilities.fillna(0.0)
    
    print()
    if return_probabilities:
        return predictions, probabilities
    else:
        return predictions

###############################################################################

def evaluate_performance(labels, predictions):
    """
    Prints several summary metrics of classification performance.
    
    Parameters
    ----------
    labels : pandas.Series
        Shape [???,]. Must be indexable by `predictions.index.levels[-1]`.
    predictions : pandas.Series
        Shape [[n_folds, fold_size],] (i.e. a MultiIndex over folds).
    """
    # samples = Panel [n_samples, n_observations, n_features]
    
    # Easier to convert to string and let sklearn sort a common coding
    # Alternative is unioning / `cat.set_categories()`, but not completely
    # trivial, and then sklearn would print the codes in the report anyway
    predictions = utils.to_categorical(predictions, to_string=True)
    # Not strictly necessary, but throws error on missing entries
    labels = labels[predictions.index.levels[-1]]
    labels = utils.to_categorical(labels, to_string=True)
    labels = labels.reindex(predictions.index, level=-1)
    
    print("Evaluating classification performance...")
    print("No. of predictions: {:d}".format(len(predictions)))
    print("No. of unique observations: {:d}"
          .format(len(predictions.index.levels[-1])))
    print("No. of classes: {:d}".format(len(labels.cat.categories)))
    print("Classes: {}".format(", ".join(map(str, labels.cat.categories))))
    print("N.B. metrics are calculated over all predictions, not within folds.")
    print()
    
    # Note that computing the metric over all folds, rather than averaging
    # within-fold metrics, is somewhat unusual. See e.g.:
    # https://datascience.stackexchange.com/q/53773
    # https://doi.org/10.1145/1882471.1882479
    # However, this approach does have some advantages: it makes running into
    # degenerate metrics when performing e.g. leave-one-out CV much less likely
    # `sklearn.metrics.f1_score(y_true=[0], y_pred=[0])`
    # Furthermore, we make sure to assess this via permutation tests, which
    # means we should still control the error rate.
    # TODO: add to README?
    
    # Balanced accuracy
    acc = sklearn.metrics.balanced_accuracy_score(labels, predictions)
    print("Balanced accuracy: {:.1f}%".format(100.0 * acc))
    print("Null: {:.1f}%".format(100.0 * (1.0 / len(labels.cat.categories))))
    #for class_label in labels.cat.categories:
    #    acc = sklearn.metrics.balanced_accuracy_score(
    #            labels == class_label, predictions == class_label)
    #    print("{}: {:.2f}".format(class_label, acc))
    print()
    
    # Other metrics
    print(sklearn.metrics.classification_report(labels, predictions))
    print(pd.crosstab(
            labels, predictions,
            rownames=['True'], colnames=['Predicted'], margins=True))
    print()
    
    return

###############################################################################

def compute_null_predictions(
        data, labels, classifier, parameters={},
        *, n_perms=1000, **kwargs):
    """
    Runs `predict()` on permuted data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Shape [n_observations, n_features].
    labels : pandas.Series
        Shape [???,]. Must be indexable by `data.index`.
    classifier
        A classifier (or pipeline) conforming to the sklearn interface. See
        `get_default_classifier()`.
    parameters : dict, optional
        A set of hyperparameters to tune, conforming to the sklearn
        `GridSearchCV` interface.
    n_perms : int, optional
        Number of permutations to run.
    **kwargs : optional
        All other arguments are passed to `predict()`.
    
    Returns
    -------
    predictions : list
        Length [n_perms+1,]. Each item is a [labels, output] pair, with
        `predictions[0]` being the unshuffled result.
    """
    
    print("Building null distributions for classifier performance...")
    kwargs['verbose'] = False
    Result = collections.namedtuple('Result', ['labels', 'predictions'])
    
    predictions = []
    for n in range(n_perms+1):
        permuted_labels = labels.copy()
        if n != 0:
            permuted_labels[:] = np.random.permutation(labels)
        prediction = predict(
                data, permuted_labels, classifier, parameters, **kwargs)
        predictions.append(Result(permuted_labels, prediction))
        print("Finished permutation {} of {}".format(n, n_perms))
    
    print()
    return predictions

###############################################################################

def evaluate_significance(
        null_predictions,
        *, metric=sklearn.metrics.balanced_accuracy_score):
    """
    Prints several summary metrics of classification performance.
    
    Parameters
    ----------
    null_predictions : list
        Length [n_perms+1,]. Each item is a [labels, prediction] pair, with
        `predictions[0]` being the unshuffled result (typically the output of
        `compute_null_predictions()`).
    metric : optional
        Function conforming to the `sklearn.metrics` interface.
    
    Returns
    -------
    metrics : np.array
        `metric` calculated for every item in `null_predictions`.
    """
    
    metrics = []
    for labels, predictions in null_predictions:
        # Reorder for sklearn
        # Easier to convert to string and let sklearn sort a common coding
        predictions = utils.to_categorical(predictions, to_string=True)
        # Not strictly necessary, but throws error on missing entries
        labels = labels[predictions.index.levels[-1]]
        labels = utils.to_categorical(labels, to_string=True)
        labels = labels.reindex(predictions.index, level=-1)
        
        metrics.append(metric(labels, predictions))
    
    metrics = np.asarray(metrics)
    
    # Summarise
    print("True accuracy:            {: .2f}".format(metrics[0]))
    print("Null accuracy [+/- s.d.]: {: .2f} [+/- {:.2f}]".format(
            np.mean(metrics[1:]), np.std(metrics[1:])))
    print("Approx (2.5%, 97.5%) CI:  {: .2f}, {:.2f}".format(
            np.percentile(metrics[1:], 2.5), np.percentile(metrics[1:], 97.5)))
    # Include true in permutation distribution
    # Phipson & Smyth, 2010: https://doi.org/10.2202/1544-6115.1585
    # https://stats.stackexchange.com/a/112352
    k = np.sum(metrics >= metrics[0])
    n = len(metrics)
    print("p(True > Null) [95% CI]:  {: .3f} [{:.2e}, {:.2e}]".format(
            k / n, *scipy.stats.beta.interval(0.95, k, n-k)))
    print()
    
    return metrics

###############################################################################

def explain_classifier(
        data, labels, classifier, parameters={},
        *, confounds=None, demean_confounds=True,
        l1_reg='bic', **kwargs):
    """
    Generates an interpretation of the classifier output.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Shape [n_observations, n_features].
    labels : pandas.Series
        Shape [???,]. Must be indexable by `data.index`.
    classifier
        A classifier (or pipeline) conforming to the sklearn interface, which
        must also provide `predict_proba()`.
    parameters : dict, optional
        A set of hyperparameters to tune, conforming to the sklearn
        `GridSearchCV` interface.
    
    Returns
    -------
    explanation
        This contains the data, fitted classifier, and the explanation in the
        form of a shap `KernelExplainer` and a set of SHAP values.
    
    Other Parameters
    ----------------
    confounds : pandas.DataFrame, optional
        Shape [???, n_confounds]. Must be indexable by `data.index`.
    demean_confounds : bool, optional
        If `True`, confounds are normalised along the features axis.
    l1_reg : optional
        Passed to `shap.KernelExplainer.shap_values()`.
    **kwargs
        All other keyword args are passed to `optimise_classifier()` to
        modify the inner cross-validation loop for hyperparameter tuning.
    """
    
    print("Generating a classifier explanation...")
    
    data, labels, confounds = utils.sanitise_inputs(data, labels, confounds)
    
    # Preprocess data
    data = data.apply(sklearn.preprocessing.scale)
    if confounds is not None:
        data = utils.remove_confounds(data, confounds, demean_confounds)
    
    # Fit to data, and tune hyperparameters
    clf = optimise_classifier(
            data, labels.cat.codes, classifier, parameters, **kwargs)
    print("Classifier parameters optimised.")
    print(clf.best_params_)
    clf = clf.best_estimator_
    
    explainer = shap.KernelExplainer(
            clf.predict_proba, data,
            link='logit', keep_index=True)
    shap_values = explainer.shap_values(
            data, l1_reg=l1_reg)
    print("SHAP values generated.")
    
    # And wrap all the results up
    explanation = types.SimpleNamespace()
    explanation.data           = data
    explanation.labels         = labels
    explanation.classifier     = clf
    explanation.clf_categories = labels.cat.categories[clf.classes_]
    explanation.explainer      = explainer
    explanation.shap_values    = shap_values
    
    print()
    return explanation

###############################################################################
# Helper functions
###############################################################################

def optimise_classifier(
        data, labels, classifier, parameters,
        *, n_splits=10, n_repeats=3, scoring='balanced_accuracy'):
    """
    Fits a classifier to the data, including hyperparameter tuning.
    
    Parameters
    ----------
    data : array-like
        Shape [n_observations, n_features].
    labels : array-like
        Shape [n_observations,].
    classifier
        A classifier (or pipeline) conforming to the sklearn interface.
    parameters : dict
        A set of hyperparameters to tune, conforming to the sklearn
        `GridSearchCV` interface.
    
    Returns
    -------
    optimiser : sklearn.model_selection.GridSearchCV
        The results of the cross-validation. The trained best classifier can
        be extracted with `optimiser.best_estimator_`, or via e.g.
        `optimiser.fit()`.
    
    Other Parameters
    ----------------
    n_splits : int, optional
    n_repeats : int, optional
        Parameters for `sklearn.model_selection.RepeatedStratifiedKFold`.
    scoring : optional
        The scoring method passed to `sklearn.model_selection.GridSearchCV`.
    """
    
    # Try to keep at least one class in each fold
    min_count = min(np.unique(labels, return_counts=True)[1])
    if min_count < n_splits:
        warnings.warn("optimise_classifier(): adjusting `n_splits`. More splits requested than observations in smallest class.", RuntimeWarning)
    n_splits = max(2, min(n_splits, min_count))
    cv_iter = sklearn.model_selection.RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats)
    
    optimiser = sklearn.model_selection.GridSearchCV(
        classifier, parameters, cv=cv_iter,
        scoring=scoring, refit=True, n_jobs=-1)
    
    with sklearn.utils.parallel_backend('threading'):
        optimiser.fit(data, labels)
    
    return optimiser

###############################################################################
