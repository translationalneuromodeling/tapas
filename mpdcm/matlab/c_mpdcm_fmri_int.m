%% [ y ] = c_mpdcm_fmri_int(u, theta, ptheta)
% Integrates several dcms with inputs u, parameters theta, and ptheta.
%
% Input:
% u     -- Two dimensional cell array of inputs to the dynamic system. u should 
%   Mx1 dimensions, where M is the total number of different DCM's to be 
%   integrated. Each cell should be a double, non sparse, real matrix of
%   dimensions NxD, where N is the number of regions and D is the number of 
%   data points. All cells in u should have the same dimensions, and should
%   be padded with nan if necessary.
%
% theta -- Two dimensional cell array of parameters to the dynamic system.
%   theta should have MxN dimensions. Each cell should be a structure as 
%   specified bellow. For a cell mxn, the system will be integrated with 
%   inputs u{m, 1} and parameters theta{m, n}. 
%
% ptheta -- A structure array as specified bellow. All systems will be 
%   integrated with the same values.
%
% Output:
% y     -- An MxN cell array containing double, non sparse matrices. Cell
%   y{m, n} corresponds to a DCM integrated with input u{m, 1}, parameters
%   theta{m, n} and ptheta.
%
% This implementation is intended two allow for efficient
% hierarchical inferece and computation of free energies using 
% thermodynamic integration. In the first case, different sets of inputs
% with diffent parameters can be specified along the first dimension of u.
% In the latter case, a collection of systems can be integrated with a 
% single input. That is useful while performing mcmc at differnt
% temperatures.
%
% This function doesn't verify it's input. Therefore a failure to comply 
% with the API will most likely produce a segmentation error. In order to
% check your input see:
%
%       mpdcm_fmri_int_checkinput
%
% theta should be an structure containing the following fields:
%
% theta.dimx -- Number of nodes in the network.
% theta.dimu -- Number of inputs of u.
%
% theta.A -- Double matrix (non sparse, no imaginary) of dimensions 
%      theta.dimx X theta.dimx
% theta.B -- Cell array of total size theta.dim_u, where each cell is a 
%       matrix of size theta.dimx X theta.dimx.
% theta.C -- Double matrix of size theta.dimx X theta.dimu
% theta.x0 -- A double matrix of size theta.dimx X 5. This corresponds to
%       the initial conditions of the system.
%
% theta.K -- Double
% theta.V0 -- Double
% theta.E0 -- Double
% theta.k1 -- Double
% theta.k2 -- Double 
% theta.k3 -- Double
% theta.alpha -- Double
% theta.gamma -- Double
% theta.tau -- Double
%
% ptheta should be structure with parameters
%
% ptheta.dt -- Subsampling precision. Should be double between 0 and 1. 
%       ceil(1/ptheta.dt) integration steps will be performed between data
%       points. Input u are theated as boxcart functions accross integration
%       steps.
%
% For a working example of how to use this library see:
%
%      test_mpdcm_fmri_int.m
%
% In the fuction test_mpdcm_fmri_int you will find an example.
%

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%

