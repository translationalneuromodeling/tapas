%% [ nParameters, idxParamsInf, idxSelfCon ] = tapas_huge_count_params( DcmInfo )
%
% Calculate number of parameters and parameter indices for DCM.
%
% INPUT:
%       DcmInfo - struct containing DCM model specification and BOLD time
%                 series.
%
% OUTPUT:
%       nParameters  - 2-by-3 array containing parameter counts
%       idxParamsInf - indices of parameters being inferred
%       idxSelfCon   - indices of self connections 
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
function [ nParameters, idxParamsInf, idxSelfCon ] = tapas_huge_count_params( DcmInfo )
% length of full DCM parameter vector
nDcmParametersCon = DcmInfo.nStates*(DcmInfo.nStates + ... % A
                             DcmInfo.nStates*DcmInfo.nInputs + ... % B
                             DcmInfo.nInputs + ... % C
                             DcmInfo.nStates*DcmInfo.nStates); % D
nDcmParametersHem = DcmInfo.nStates*3; % hem
nDcmParametersAll = nDcmParametersCon + nDcmParametersHem;

% number of parameters to be inferred %%% dim(theta_c) dim(theta_h)
idxParamsInf = (1:nDcmParametersAll).';
idxParamsInf(DcmInfo.noConnectionIndicator) = 0;
idxParamsInf(end-DcmInfo.nStates+2:end) = 0; % only first epsilon is used
idxParamsInf = idxParamsInf(idxParamsInf~=0);

nDcmParamsInfCon = nnz(idxParamsInf<=nDcmParametersCon); %%% d_c
% nDcmParamsInfCon = DcmInfo.nConnections; % alternative
nDcmParamsInfHem = nnz(idxParamsInf>nDcmParametersCon); %%% d_h
nDcmParamsInfAll = length(idxParamsInf); %%% d

% indices of diagonal elements of A (self connections)
idxSelfCon = find(eye(size(DcmInfo.adjacencyA)));
idxSelfCon = intersect(idxParamsInf,idxSelfCon);

nParameters = [nDcmParametersCon,nDcmParametersHem,nDcmParametersAll;...
               nDcmParamsInfCon,nDcmParamsInfHem,nDcmParamsInfAll;];


end

