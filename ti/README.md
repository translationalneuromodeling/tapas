# README

aponteeduardo@gmail.com
copyright (C) 2016

This is a general routine to estimate behavioral models with the following
notation.

y           Data
u           Experimental design
theta       Model parameters
ptheta      Model and priors
htheta      Parameter of the updating method


## ptheta

ptheta is a definition of the model. A model is defined by a likelihood 
and prior function. Both functions are defined as

ptheta.llh -> Likelihood
ptheta.lpp -> Prior

Both distribution should have the following signature

l = llh(y, u, theta, ptheta)
p = lpp(theta, ptheta)


