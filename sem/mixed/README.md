# README

aponteeduardo@gmail.com
copyright (C) 2017

This is the interface a mixed multivariate linear model with diagonal 
covariances.

In order to specify the mixed effects the parameter

ptheta.x = X; % Design matrix
ptheta.mixed = mixed; % Mixed effects.

The number of grouping factors is size(mixed, 2). 
The first dimension of mixed should be equal to the number of regressors
used in the design or size(X, 2); It is assumed to be a logical array,
if not it will be transformed into a logical. The ones in a column 
designate membeship to a cluster or group and the prior of this column
will be estimated as 

(sum(x(:, 1)) * mean(b(x(:, i)), 1) * precision + 
    prior mean * prior precision) / 
        (sum(x(:, i)) * precision + prior precision)

Note that only the mean will be updated. A detail of the model can be 
found in [].
