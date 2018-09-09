%% [response,x,s,f,v,q] = tapas_huge_bold( packedParameters, DcmInfo, iSubject )
% 
% Compute the predicted response for the DCM-FMRI model using numerical
% integration of differential equations.
%
% INPUT:
%       packedParameters - current value of parameters
%       DcmInfo          - struct containing DCM model specification and
%                          BOLD time series.
%       iSubject         - subject index
% 
% OUTPUT:
%       response - matrix of predicted response for each region
%                  (column-wise) 
%       x        - time series of neuronal states
%       s        - time series of vasodilatory signal 
%       f1       - time series of flow
%       v1       - time series of blood volume
%       q1       - time series of deoxyhemoglobin content.
%           
% REFERENCE:
%   Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A.
%   Robinson, Karl J. Friston (2007). Comparing hemodynamic models with
%   DCM. NeuroImage, 38: 387-401
% 
% https://doi.org/10.1016/j.neuroimage.2007.07.040
%

%
% Author: Sudhir Shankar Raman
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
%%
function [response,x,s,f,v,q] = tapas_huge_bold( packedParameters, DcmInfo, iSubject )

paramList = [DcmInfo.timeStep(iSubject) DcmInfo.nTime(iSubject) ...
    DcmInfo.nStates DcmInfo.nInputs iSubject DcmInfo.dcmTypeB ...
    DcmInfo.dcmTypeD];

[packedParameters] = tapas_huge_pack_params(packedParameters,paramList);

%----------------------------------------------------------------
% unpack parameters 
HEM_LIST = DcmInfo.hemParam.listHem; % kappa, tau, epsilon
% transpose of the original A
A = (packedParameters{1}.*DcmInfo.adjacencyA)';
% transpose of the original C
C = (packedParameters{2}.*DcmInfo.adjacencyC)';
% For Lorenz Dataset Analysis
C = C./DcmInfo.hemParam.scaleC;
B = (packedParameters{3}); % original B
D = packedParameters{4};
tau = HEM_LIST(2)*exp(packedParameters{5}(1,:));
kappa = HEM_LIST(1)*exp(packedParameters{5}(2,:));
epsilon = HEM_LIST(3)*exp(packedParameters{5}(3,:));
epsilon = epsilon(1)*ones(1, DcmInfo.nStates);
% estimated region-specific ratios of intra- to extra-vascular signal 
%--------------------------------------------------------------------------

%------------------------------------------------------------------
%               Hemodynamic constants
% TE
echoTime = DcmInfo.hemParam.echoTime;

% resting venous volume (%) restingVenousVolume
%--------------------------------------------------------------------------
restingVenousVolume  = DcmInfo.hemParam.restingVenousVolume;
 
% slope relaxationRateSlope of intravascular relaxation rate R_iv as a
% function of oxygen  saturation S:
% R_iv = relaxationRateSlope*[(1 - S)-(1 - S0)] (Hz)
%--------------------------------------------------------------------------
relaxationRateSlope  = DcmInfo.hemParam.relaxationRateSlope;
 
% frequency offset at the outer surface of magnetized vessels (Hz) - nu0
%--------------------------------------------------------------------------
frequencyOffset = DcmInfo.hemParam.frequencyOffset;
 
% resting oxygen extraction fraction - rho
%--------------------------------------------------------------------------
oxygenExtractionFraction = DcmInfo.hemParam.oxygenExtractionFraction*...
    ones(1,paramList(3));

%-Coefficients in BOLD signal model - 
%==========================================================================
coefficientK1  = DcmInfo.hemParam.rho*frequencyOffset*echoTime*...
    oxygenExtractionFraction;
coefficientK2  = epsilon.*(relaxationRateSlope*...
    oxygenExtractionFraction*echoTime);
coefficientK3  = 1 - epsilon;

mArrayB = B.*DcmInfo.adjacencyB;
mArrayB = permute(mArrayB,[2 1 3]);

mArrayD = D.*DcmInfo.adjacencyD;
mArrayD = permute(mArrayD,[2 1 3]);

% resting oxygen extraction fraction - change value to match what SPM does
%--------------------------------------------------------------------------
% For Lorenz Dataset Analysis
oxygenExtractionFraction = DcmInfo.hemParam.oxygenExtractionFraction2*...
    ones(1,paramList(3));
%oxygenExtractionFraction = 0.32*ones(1,paramList(3));

% pre-calculation
C = DcmInfo.listU{paramList(5)}*C;

% Integrate the dynamical system
[x,s,f,v,q]  = tapas_huge_int_euler(A,C,DcmInfo.listU{paramList(5)},...
    mArrayB,mArrayD,oxygenExtractionFraction,DcmInfo.hemParam.alphainv,...
    tau,DcmInfo.hemParam.gamma,kappa,paramList);
               
% generate the responses per time point
response = restingVenousVolume*( ...
    bsxfun(@times,coefficientK1,...
        (1 - (q(DcmInfo.listResponseTimeIndices{paramList(5)},:)))) +...
    bsxfun(@times,coefficientK2,...
        (1 - (q(DcmInfo.listResponseTimeIndices{paramList(5)},:)./...
        v(DcmInfo.listResponseTimeIndices{paramList(5)},:)))) +...
    bsxfun(@times,coefficientK3,...
        (1-v(DcmInfo.listResponseTimeIndices{paramList(5)},:))));
   


