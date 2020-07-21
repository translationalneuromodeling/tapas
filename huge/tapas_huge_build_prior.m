function [ priors, DCM ] = tapas_huge_build_prior( DCM )
%   WARNING: This function is deprecated and will be removed in a future
%   version of this toolbox. Please use the new object-oriented interface
%   provided by the tapas_Huge class.
% 
% Generate values for prior parameters for HUGE. Prior mean of cluster
% centers and prior mean and covariance of hemodynamic parameters follow
% SPM convention (SPM8 r6313).
%
% INPUT:
%       DcmInfo - cell array of DCM in SPM format
%
% OUTPUT:
%       priors  - struct containing priors
%       DcmInfo - struct containing DCM model specification and BOLD time
%                 series in DcmInfo format
%
% See also tapas_Huge, tapas_Huge.estimate, tapas_huge_demo
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 

wnMsg = ['This function is deprecated and will be removed in a future ' ...
	     'version of this toolbox. Please use the new object-oriented ' ...
         'interface provided by the tapas_Huge class.'];
warning('tapas:huge:deprecated',wnMsg)

%% check input format
if isvector(DCM)&&isstruct(DCM)
    try
        DCM = {DCM(:).DCM}';
    catch
        DCM = num2cell(DCM);
    end
else
    assert(iscell(DCM),'TAPAS:HUGE:inputFormat',...
        'DCM must be cell array of DCMs in SPM format');
end

dcm = DCM{1};


%% set priors
priors = struct();
% parameter of Dirichlet prior (alpha_0 in Figure 1 of REF [1])
priors.alpha = 1;

tmp = dcm.a/64/dcm.n;
tmp = tmp - diag(diag(tmp)) - .5*eye(dcm.n);
tmp = [tmp(:); dcm.b(:)*0; dcm.c(:)*0; ...
       dcm.d(:)*0];

connectionIndicator = find([dcm.a(:);dcm.b(:);dcm.c(:);dcm.d(:)]);
% prior mean of clusters (m_0 in Figure 1 of REF [1])
priors.clustersMean = tmp(connectionIndicator)';
%  tau_0 in Figure 1 of REF [1]
priors.clustersTau = 0.1;
% degrees of freedom of inverse-Wishart prior (nu_0 in Figure 1 of REF [1])
priors.clustersDeg = max(100,1.5^length(connectionIndicator));
priors.clustersDeg = min(priors.clustersDeg,double(realmax('single')));

% scale matrix of inverse-Wishart prior (S_0 in Figure 1 of REF [1])
priors.clustersSigma = 0.01*eye(length(connectionIndicator))*...
                       (priors.clustersDeg - length(connectionIndicator) - 1);

% prior mean of hemodynamic parameters (mu_h in Figure 1 of REF [1])
priors.hemMean = zeros(1,dcm.n*2 + 1);

% prior Covariance of hemodynamic parameters(Sigma_h in Figure 1 of
% REF [1]) 
priors.hemSigma = diag(zeros(1,dcm.n*2 + 1)+exp(-6));

% prior inverse scale of observation noise (b_0 in Figure 1 of REF [1])
priors.noiseInvScale = .025;

% prior shape parameter of observation noise (a_0 in Figure 1 of REF [1])
priors.noiseShape = 1.28;


end

