# Generative Embedding

> Version: 0.1.0
>
> Author: Sam Harrison
>
> Maintainer: Sam Harrison

A Python package for classification problems, with a focus on data exploration
and visualisation. Within TAPAS, this forms a core part of the **gen**erative
em**bed**ding workflow. The package itself is a thin wrapper around
[scikit-learn](https://scikit-learn.org), with the idea being that it provides
a set of convenience functions and sensible defaults that form a suitable
starting point for more detailed investigations. For a more in depth
discussion, see the [rationale section](#rationale).

For a more detailed overview of generative embedding, the papers listed below
are good starting points. In brief, the idea is to use generative models as a
principled, interpretable method for reducing complex, high-dimensional
(typically neuroimaging) data into a low dimensional embedding space. One can
then run e.g. a classification algorithm on the reduced set of parameters, and
that is what the `genbed` package aims to help with. For the first part of the
pipeline, TAPAS itself is a good place to start for neuroimaging models.
 + Brodersen et al., "Generative Embedding for Model-Based Classification of
   fMRI Data", PLoS Computational Biology, 2011.
   DOI:&nbsp;[10.1371/journal.pcbi.1002079](https://doi.org/10.1371/journal.pcbi.1002079).
 + Stephan et al., "Computational neuroimaging strategies for single patient
   predictions", NeuroImage, 2017.
   DOI:&nbsp;[10.1016/j.neuroimage.2016.06.038](https://doi.org/10.1016/j.neuroimage.2016.06.038).

Finally, note that the package is still in a beta version, in the sense that
the API is still likely to change! If there are workflows / visualisations /
etc. that are not included but would form a useful part of the overall
classification pipeline, then please let us know.

-------------------------

### Citing `genbed`

You can include for example the following snippet in your Methods section,
along with a brief description of the classification pipeline used. See the
[main TAPAS README](../README.md) for more details on citing TAPAS itself.

> The analysis was performed using the Python genbed toolbox (version 0.1.0,
> open-source code available as part of the TAPAS software collection: [1] /
> <https://www.translationalneuromodeling.org/tapas>)

 1. Frässle, S.; Aponte, E.A.; Bollmann, S.; Brodersen, K.H.; Do, C.T.;
    Harrison, O.K.; Harrison, S.J.; Heinzle, J.; Iglesias, S.; Kasper, L.;
    Lomakina, E.I.; Mathys, C.; Müller-Schrader, M.; Pereira, I.; Petzschner,
    F.H.; Raman, S.; Schöbi, D.; Toussaint, B.; Weber, L.A.; Yao, Y.; Stephan,
    K.E.; *TAPAS: An Open-Source Software Package for Translational
    Neuromodeling and Computational Psychiatry*, Frontiers in Psychiatry 12,
    857, 2021. <https://doi.org/10.3389/fpsyt.2021.680811>

-------------------------

### Installation

We use [`conda`](https://conda.io/) to manage the Python dependencies. We can
use this to create and manage a bespoke Python environment. If this (or
Anaconda) is not already installed, then you will need to do this first.
We recommend Miniconda as this gives a simple, lightweight installation. For
more details see:
 + <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>
 + <https://docs.conda.io/en/latest/miniconda.html>
 + <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda>

Once `conda` is available, installation of the `genbed` environment should
simply be a case of using the supplied [`environment.yml`](environment.yml)
file:
```shell
cd genbed/
conda env create --file environment.yml
conda activate genbed
```

For now, we work with the local copy of the code rather than installing. That
means you either need to work in the `genbed/` directory, or use the following
to temporarily add the code to your Python path:
```python
import sys
sys.path.insert(0, '/path/to/genbed')
```

-------------------------

### Getting started

The `genbed` environment contains both
[IPython](https://ipython.readthedocs.io/en/stable/interactive/tutorial.html)
and [Jupyter](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html)
for running code interactively. See the links for more documentation.

The easiest place to start is probably the included Jupyter notebook:
[`doc/ExampleAnalysis.ipynb`](doc/ExampleAnalysis.ipynb). To run locally, use
the following and then navigate to the notebook itself:
```shell
conda activate genbed
jupyter notebook
```

The rest of this README provides a more detailed overview of the available
functionality, and how to build a workflow suitable for your own data.

-------------------------

### Data organisation

The whole toolbox is based around the predictors (`data`) and classes
(`labels`), and these are represented as a pandas `DataFrame` and `Series`
respectively. This makes it easier to keep track of metadata like feature
names, as well as making it *much* easier to keep observations correctly linked
with descriptive index names. We strongly recommend explicitly labelling inputs
(e.g. using a `subject_id` to index observations).

-------------------------

### Typical workflow

The typical workflow is concentrates on three key areas:
 + Data exploration and visualisation.
 + Classification and performance assessment.
 + Post-hoc interpretation of classifier performance.

A complete example, including typical outputs, can be found as a Jupyter
notebook: [`doc/ExampleAnalysis.ipynb`](doc/ExampleAnalysis.ipynb).

```python
import pandas as pd
import matplotlib, matplotlib.pyplot as plt

import genbed.classification as gclass
import genbed.statistics as gstats
import genbed.visualisation as gvis

# Load data
data = pd.read_csv('path/to/data.csv', index_col='subject_id') # header=0

# Load labels/classes
labels = pd.read_csv('path/to/labels.csv', index_col='subject_id', squeeze=True)
# Alternatively, if bundled with `data`
labels = data['labels']; data = data.drop(columns='labels')
# Potentially useful:
#   `labels = labels.astype('category')`
#   `labels = labels.cat.rename_categories(...)`
#   `labels = labels.cat.reorder_categories(...)`

# All functions allow the removal of a set of confounds via linear regression
confounds = pd.read_csv('path/to/confounds.csv', index_col='subject_id')

# Plot data
gvis.visualise_data(data, labels, confounds=confounds)
gvis.visualise_manifold(data, labels, confounds=confounds)
plt.show()

# Univariate tests for group differences
gstats.run_tests(data, labels, confounds=confounds)

# Run classification
# `predict()` performs nested cross-validation
classifier, parameters = gclass.get_default_classifier()
predictions, probabilities = gclass.predict(
        data, labels, classifier, parameters,
        confounds=confounds, return_probabilities=True)
gclass.evaluate_performance(labels, predictions)

# Plot classifier output
gvis.visualise_performance(labels, predictions, probabilities)
plt.show()

# Check significance via permutation testing
# N.B. possibly slow - more permutations will be needed in reality!
null_predictions = gclass.compute_null_predictions(
        data, labels, classifier, parameters,
        confounds=confounds, n_perms=100)
metrics = gclass.evaluate_significance(null_predictions)
gvis.visualise_null_metrics(metrics)

# Interpret classifier output
explanation = gclass.explain_classifier(
        data, labels, classifier, parameters,
        confounds=confounds)
gvis.visualise_explanation(explanation)
plt.show()
```

Note that for small datasets (and given an appropriate classifier), it may well
be useful to change to a continuous scoring function for the inner
cross-validation loop (e.g. log loss, Brier score) as demonstrated below.
Similarly, it may also be necessary to increase the number of times both the
inner and outer loops are repeated if there is obvious run-to-run variability.

-------------------------

### Preprocessing

The toolbox as a whole has to make several assumptions regarding the data
itself. We list the key ones explicitly here for completeness, though one
expects that they will be met automatically for most data.


##### Data format

All data and confounds should be represented numerically (or as a type with a
straightforward conversion to float). For e.g. categorical data, use the
functionality in [`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/preprocessing.html)
to re-encode the data. Similarly, for missing data use [`sklearn.impute`](https://scikit-learn.org/stable/modules/impute.html).


##### Confounds

All functions that take data also allow removal of a set of confounds via
linear regression. By default, the confounds are demeaned as this is almost
always the expected and required behaviour. In the very rare cases where this
is not desirable, it is possible to explicitly disable it. There is a further
discussion of this issue in the [rationale section](#rationale).

Finally, a note on categorical confounds: if using a one-hot encoding scheme to
represent the confounds then these become perfectly colinear after demeaning.
To avoid problems, simply drop one of the categories by using e.g.
`sklearn.preprocessing.OneHotEncoder(drop='first')`.

-------------------------

### Advanced usage

The core functions are designed to be flexible and allow modular components
from the `sklearn` interface to be interchanged. For example:

```python
import numpy as np
import sklearn
import sklearn.preprocessing, sklearn.decomposition, sklearn.metrics
import sklearn.model_selection, sklearn.svm, sklearn.pipeline

# --------------------

# Use a different manifold learning algorithm
gvis.visualise_manifold(
        data, labels, confounds=confounds,
        manifold=sklearn.decomposition.PCA())
plt.show()

# --------------------

# Turn on dimensionality reduction and feature selection for classification
#  - `feature_reduction` reduces the dimensionality by combining features
#    (so is similar in spirit to feature agglomeration)
#  - `feature_selection` performs univariate selection (on the possibly
#    reduced feature set)
classifier, parameters = gclass.get_default_classifier(
        feature_reduction=True, feature_selection=True)
predictions = gclass.predict(
        data, labels, classifier, parameters,
        confounds=confounds)

# --------------------

# Change the behaviour of the outer cross-validation loop
cv_iter = sklearn.model_selection.RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3)
predictions = gclass.predict(
        data, labels, classifier, parameters,
        confounds=confounds, cv_iter=cv_iter)

# Change the behaviour of the inner cross-validation loop
predictions = gclass.predict(
        data, labels, classifier, parameters,
        confounds=confounds, inner_cv_kwargs=dict(
            n_splits=5, n_repeats=5, scoring='neg_log_loss'))

# --------------------

# Change the classifier
classifier = sklearn.svm.SVC(
        kernel='linear', class_weight='balanced',
        decision_function_shape='ovr', max_iter=1.0e3)
parameters = {
        'C': np.geomspace(1.0e-3, 1.0e3, 15)}
print(classifier, parameters)

# Fit a more complex pipeline
classifier = sklearn.pipeline.Pipeline([
        ('qt', sklearn.preprocessing.QuantileTransformer(n_quantiles=10)),
        ('clf', classifier)])
parameters = {'clf__'+name: values for name, values in parameters.items()}
print(classifier, parameters)

# This can even be combined with feature reduction/selection
classifier, parameters = gclass.get_default_classifier(
        classifier, parameters,
        feature_reduction=True, feature_selection=True)
print(classifier, parameters)

predictions = gclass.predict(
        data, labels, classifier, parameters,
        confounds=confounds)

# --------------------

# Change the metric used when calculating null distributions
# Return probabilities and use them to calculate negative log-loss
null_predictions = gclass.compute_null_predictions(
        data, labels, classifier, parameters,
        confounds=confounds, n_perms=100,
        return_probabilities=True)
null_probabilities = [[l, probs] for l, [preds, probs] in null_predictions]
metrics = gclass.evaluate_significance(
        null_probabilities,
        metric=lambda x,y: -sklearn.metrics.log_loss(x,y))
gvis.visualise_null_metrics(metrics)

# --------------------
```

-------------------------

### Rationale


##### Default classifier

This is logistic regression for several reasons:
 + Firstly, the simple linear model wards against overfitting and is
   (relatively) simple to interpret.
 + Secondly, the well calibrated class probabilities are useful for
   visualisation of the classifier outputs \[1\].
 + Thirdly, the generalisation to the multi-class case is straightforward via
   the multinomial / cross-entropy loss.
 + Finally, while it does not have the max-margin property of SVMs, it is a
   large-margin classifier \[2\].

\[1\]: Niculescu-Mizil & Caruana, "Predicting good probabilities with
supervised learning", ICML, 2005.
DOI:&nbsp;[10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430)

\[2\]: Lecun et al., "A Tutorial on Energy-Based Learning", Predicting
structured data, 2006.
[Link](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)


##### Cross-validation

By default, we use repeated k-fold cross-validation for both model selection
(i.e. hyperparameter tuning) and model evaluation (i.e. estimating performance
on unseen data).

Sources of variance:
 + Finite sample variance: we only ever have a fixed number of samples to work
   with, and this limits how confident we can be in our performance / parameter
   estimates. If our classifier achieves a 'true' accuracy of 70%, and we fit
   it on 100 *new* samples from the population, then we would expect anything
   from 60 to 80 correct classifications (≈95% binomial confidence interval).
   Note that we don't 'see' this in our data, as we are not able to resample
   from the population (except in simulations). Rather, we have to acknowledge
   that we can only learn so much from limited data.

 + Model instability variance: models can give different predictions for the
   same sample. There may be inherent randomness, where the prediction will
   differ even on the same training set (this could be due to randomness in the
   algorithm itself, or in e.g. the cross-validation procedure if
   hyperparameters need to be tuned). The algorithm could also be sensitive to
   perturbations in the training set, such that different folds give different
   predictions for the same sample. In contrast to the finite sample variance,
   this is something we can observe directly.

Sources of bias:
 + Diminished training size: if we leave out 10% of our data for validation,
   then we are really estimating the performance of a model trained on 90% of
   the available data. For small datasets, it is possible that adding more
   samples gives an appreciable increase in performance. This makes
   cross-validation pessimistically biased.

 + Overfitting during model selection: if we throw too many options at nested
   cross-validation, then it can overfit during the model selection step
   \[1,2\].  Again, this can pessimistically bias performance estimates.
   \[Aside: perhaps we should do model averaging / significance testing in the
   inner loop?\]

Our use of repeated k-fold is motivated by several observations. Compared to
leave-on-out (LOO) cross-validation (CV), it allows the quantification of (or
at least averaging over) the model instability variance. K-fold CV will have a
stronger pessimistic bias due to the smaller training sets, but potentially a
lower variance (though see [here](https://stats.stackexchange.com/q/61783) and
[here](https://stats.stackexchange.com/q/280665)&mdash;but it's not always
clear if this refers to within- or out-of-sample errors). In practice, setting
k≈10 seems to be a reasonable compromise. Finally, for large datasets it is
significantly cheaper computationally.

\[1\]: Cawley & Talbot, "On Over-fitting in Model Selection and Subsequent
Selection Bias in Performance Evaluation", JMLR, 2004.
[Link](http://jmlr.org/papers/v11/cawley10a.html)

\[2\]: Varoquaux et al., "Assessing and tuning brain decoders:
Cross-validation, caveats, and guidelines", NeuroImage, 2017.
DOI:&nbsp;[10.1016/j.neuroimage.2016.10.038](https://doi.org/10.1016/j.neuroimage.2016.10.038)


##### Confound correction

We use the 'cross-validated confound regression' approach of Snoek et al.
\[1\]. This regresses out confounds within each fold, as other simpler methods
may lead to biased results.

> 'When you win the toss — bat. If you are in doubt, think about it — then bat.
> If you have very big doubts, consult a colleague — then bat.'
>
> W. G. Grace

This advice applies equally well to demeaning confounds as to winning the toss
and batting. As discussed earlier, `genbed` demeans confounds by default and we
give a more intuitive overview in the
[`doc/Confounds.ipynb`](doc/Confounds.ipynb) Jupyter Notebook.

\[1\]: Snoek et al., "How to control for confounds in decoding analyses of
neuroimaging data", NeuroImage, 2019.
DOI:&nbsp;[10.1016/j.neuroimage.2018.09.074](https://doi.org/10.1016/j.neuroimage.2018.09.074)


##### Significance testing

For small sample sizes the error bars on classification accuracies can be large
(easily +/-10%). Gaël Varoquaux has an excellent discussion of the estimation
of cross-validation error bars in neuroimaging \[1\] (and also e.g. \[2\]).

Here, all significance testing is done via permutation tests. While it is
possible to use parametric tests (e.g.
[binomial confidence intervals](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval),
[McNemar's test](https://en.wikipedia.org/wiki/McNemar%27s_test), and their
extensions to other scores \[3,4\]), these tests underestimate the variance
and are therefore overconfident \[1,5,6\].

Intuitively, there are two key problems with the above parametric tests. The
first is that they assume that each correct / incorrect classification is an
independent observation. However, they are correlated: errors within folds are
linked because they share a common training set / classifier, and across folds
because the different training sets share a large number of observations \[5\].
The second is that they make the assumption that the underlying true
classification accuracy is the same for each observation. This does not hold
either, as each fold trains a different classifier, and these will have
differing accuracies. See [here](https://stats.stackexchange.com/q/88183) for a
good discussion of these issues.

\[1\]: Varoquaux, "Cross-validation failure: Small sample sizes lead to large
error bars", NeuroImage, 2018.
DOI:&nbsp;[10.1016/j.neuroimage.2017.06.061](https://doi.org/10.1016/j.neuroimage.2017.06.061)

\[2\]: Combrisson & Jerbi, "Exceeding chance level by chance: The caveat of
theoretical chance levels in brain signal classification and statistical
assessment of decoding accuracy", Journal of Neuroscience Methods, 2015.
DOI:&nbsp;[10.1016/j.jneumeth.2015.01.010](https://doi.org/10.1016/j.jneumeth.2015.01.010)

\[3\]: Brodersen et al., "The Balanced Accuracy and Its Posterior
Distribution", ICPR, 2010.
DOI:&nbsp;[10.1109/ICPR.2010.764](https://doi.org/10.1109/ICPR.2010.764)

\[4\]: Carillo et al., "Probabilistic Performance Evaluation for Multiclass
Classification Using the Posterior Balanced Accuracy", ROBOT, 2013.
DOI:&nbsp;[10.1007/978-3-319-03413-3_25](https://doi.org/10.1007/978-3-319-03413-3_25)

\[5\]: Bengio & Grandvalet, "No Unbiased Estimator of the Variance of K-Fold
Cross-Validation", JMLR, 2004.
[Link](http://jmlr.org/papers/v5/grandvalet04a.html)

\[6\]: Grandvalet & Bengio, "Hypothesis Testing for Cross-Validation", 2006.
[Link](https://pdfs.semanticscholar.org/07bb/9e2c07ddcc05ad7338f7f305d4f109d07b9b.pdf)


##### Classifier interpretation

We use [SHAP](https://github.com/slundberg/shap) to inspect classifier outputs.
This unifies several suggested approaches, as is explained in the NeurIPS paper
\[1\].

For linear classifiers, there are exact solutions to this (see e.g. Haufe et
al. \[2\] for a neuroimaging-specific example), but we use SHAP to remain
agnostic to the choice of classifier.

\[1\]: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
NeurIPS, 2017.
[Link](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

\[2\]: Haufe et al., "On the interpretation of weight vectors of linear models
in multivariate neuroimaging", NeuroImage, 2014.
DOI:&nbsp;[10.1016/j.neuroimage.2013.10.067](https://doi.org/10.1016/j.neuroimage.2013.10.067)

-------------------------
