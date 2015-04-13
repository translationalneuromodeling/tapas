# README

aponteeduardo@gmail.com
copyright (C) 2015

## Introduction

mpdcm is a toolbox for efficient integration of the dynamical system defined
by DCM (see citations) on GPU architectures. Besides the core code, several 
routines for statistical inference are defined as part of our toolbox, 
including a map estimator, a mle estimator, variational Bayes hierarchical 
inference, thermodynamic integration, and population MCMC. All this tools 
are documented below.

## Compilation

Currently only LINUX is supported for compilation. We have compiled in a few
different platforms successfully. Please refer to the doc/test.tx

For compilation you require a matlab running environment, a cuda installation
and the supported compiler. If your standard compiler is not compatible with
your cuda installation, then, although it''s possible to compile, it might
require some manual work. For compilation you should use

 cd src
 ./configure
 make

If your cuda installation happens to be in a non-standard location, or you
would like to use a version of cuda installed in your computer but not the 
standard one use

 ./configure --with-cuda=<LOCATION OF YOUR CUDA INSTALLATION>

For example

 ./configure --with-cuda=/cluster/apps/cuda/4.2.9/

## Initilization

Simply execute the file on the top folder startup()

## MAP estimator

## MLE estimator

## VM hierarchical inference

## Thermodynamic integration

## Population MCMC
