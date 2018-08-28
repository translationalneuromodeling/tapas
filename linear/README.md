# README

aponteeduardo@gmail.com
copyright (C) 2017

This are mostly hierarchical priors design to be combines with different 
models.

## dlinear

This is a hierarchical prior that only models the population mean assuming
a precision matrix of the form p\*I. This uses the standar Gamma gaussian
model.

## mdlinear

This is a multivariate hierarchical prior that models linear effects. Every
parameters is assumend to have its own precision. Thus for a model with 
parameter gamma:

gamma = X\*b + e

Where the variance of e is of the for p\*I.

## Mixedlinear

This is implemented for a mixed effects model in which a subset of the
factors are grouped toghether. This model is not particularly general and it
is not design to be so. For a multivariate linear model, it is assumed hat
some regressors have a shared mean. The mean (plus a prior expectation) is
estimated.
