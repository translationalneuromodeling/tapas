TAPAS Variational Bayesian linear regression
Release Tapas 1.4.0.0 for MATLAB, November 2014

************************************************************************
Copyright (C) 2012-2013 Kay H. Brodersen brodersen@biomed.ee.ethz.ch
Translational Neuromodeling Unit (TNU)
University of Zurich and ETH Zurich
------------------------------------------------------------------------

Software note

This software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this software. If not, see http://www.gnu.org/licenses/.

Overview
The conceptual and practical limitations of classical multiple linear regression models of the form can be resolved naturally in a Bayesian framework. Unless based on an overly simplistic parameterization, however, exact inference in Bayesian regression models is analytically intractable. This problem can be overcome using methods for approximate inference. Here, we provide a simple implementation of variational Bayesian inference for a fully Bayesian multiple linear regression model. The code is as easy to use as classical regression implementations, such as regress(), and requires no prerequisites other than MATLAB and the MATLAB Statistics Toolbox.

Model
We consider a multiple linear regression model with a shrinkage prior on the regression coefficients. This model is a generalization of the model illustrated in: Bishop, C.M., Pattern
Recognition and Machine Learning (2005), Springer, pp. 486–490. We wish to infer on the coefficients , their precision , and the noise precision . There is no analytical posterior. We therefore seek a variational approximation. Please refer to the Readme.pdf file for more details. 

DOWNLOADS & RELEASE INFORMATION

- Current Release:  Solved a log regarding the update of the lambda term reported in  http://sympa.ethz.ch/sympa/arc/tapas/2014-10/msg00008.html. Additionally, the Gram matrix is now precomputed - aponteeduardo@gmail.com


