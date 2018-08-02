%% dcm_vb_parameters_default
% 
% script for setting default parameter values for variational Basian
% inversion of HUGE
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2018 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <http://www.gnu.org/licenses/>.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is in an early stage of
% development. Considerable changes are planned for future releases. For
% support please refer to:
% https://github.com/translationalneuromodeling/tapas/issues
%
%% variational parameters
% stopping criterion: minimum increase in free energy
epsEnergy = 1e-5;
% stopping cirterion: maximum number of iterations
nIterations = 1e3;
% number of clusters
nClusters = 2;

%% computational and technical parameters
% method for calculating jacobian matrix
fnJacobian = @tapas_huge_jacobian;
% small constant to be added to the diagonal of inv(postDcmSigma) for
% numerical stability
diagConst = 1e-10;
% keep history of parameters and important auxiliary variables
bKeepTrace = false;
% keep history of response related auxiliary variables
% has no effect if bKeepTrace is false
bKeepResp = false;
bVerbose = false;


