function [m0, l0, a0, b0] = tapas_rdcm_get_prior_all(DCM)
% [m0, l0, a0, b0] = tapas_rdcm_get_prior_all(DCM)
% 
% Returns prior parameters on model parameters (theta) and noise precision
% (tau) for the full connectivity model. Necessary for the sparse version
% of rDCM.
% 
%   Input:
%   	DCM             - model structure
%
%   Output:
%       m0              - prior mean
%       l0              - prior covariance
%       a0              - prior shape parameter
%       b0              - prior rate parameter
%
 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2021 Translational Neuromodeling Unit
%                         Institute for Biomedical Engineering
%                         University of Zurich & ETH Zurich
%
% This file is part of the TAPAS rDCM Toolbox, which is released under the 
% terms of the GNU General Public License (GPL), version 3.0 or later. You
% can redistribute and/or modify the code under the terms of the GPL. For
% further see COPYING or <http://www.gnu.org/licenses/>.
% 
% Please note that this toolbox is in an early stage of development. Changes 
% are likely to occur in future releases.
% 
% ----------------------------------------------------------------------


% get the prior mean and covariance from SPM
[pE,pC] = tapas_rdcm_spm_dcm_fmri_priors(ones(size(DCM.a)),ones(size(DCM.b)),ones(size(DCM.c)),ones(size(DCM.d)));

% number of regions and inputs
[nr, nu] = size(pE.C);

% set the prior mean of endogenous parameters to zero
pE.A = zeros(size(pE.A))+diag(diag(pE.A));

% prior mean 
m0 = [pE.A reshape(pE.B,nr,nr*nu) pE.C];

% prior precision
pC.A       = 1./pC.A;
pC.B       = 1./pC.B;
pC.C       = 1./pC.C;
pC.D       = 1./pC.D;
pC.transit = 1./pC.transit;
pC.decay   = 1./pC.decay;
pC.epsilon = 1./pC.epsilon;

% prior precision
l0 = [pC.A reshape(pC.B,nr,nr*nu) pC.C];

% Setting priors on noise
a0 = 2;
b0 = 1;

end
