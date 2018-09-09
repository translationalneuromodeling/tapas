%% function [response,x,s,f,v,q] = tapas_huge_predict( dcmParamsInf, dcmParamsDefault, idxDcmParamsInf, ~, DcmInfo, iSubject)
% 
% Wrapper for the function for tapas_huge_bold.m
%
%--------------------------------------------------------------------------------------
% INPUT:
%       dcmParamsInf     - values of DCM parameters being inferred
%       dcmParamsDefault - full DCM parameters vector containing default
%                          values for remaining parameters
%       idxDcmParamsInf  - indices of DCM parameters being inferred
%       idxDiagA         - indices of self connections (dummy)
%       DcmInfo          - struct containing DCM model specification and
%                          BOLD time series.
%       iSubject         - subject index
%
%       
%---------------------------------------------------------------------------------------
% OUTPUT:
%       response - matrix of predicted response for each region
%                  (column-wise) 
%       x - time series of neuronal states
%       s - time series of vasodilatory signal 
%       f1 - time series of flow
%       v1 -  time series of blood volume
%       q1 - time series of deoxyhemoglobin content.
%           
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
function [response,x,s,f,v,q] = tapas_huge_predict( dcmParamsInf, dcmParamsDefault, idxDcmParamsInf, ~, DcmInfo, iSubject)

dcmParamsDefault(idxDcmParamsInf) = dcmParamsInf;


[response,x,s,f,v,q] = tapas_huge_bold(dcmParamsDefault,...
                                       DcmInfo,...
                                       iSubject);
response = response(:);


