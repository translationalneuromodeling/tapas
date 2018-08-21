%% [  ] = tapas_huge_generate_examples( tname )
%
% generate two synthetic example datasets based on a 2-region and a
% 3-region DCM and save them in SPM format.
% INPUT:
%       tname - filename for saving datasets
%
% REFERENCE:
% [1] Yao Y, Raman SS, Schiek M, Leff A, Frässle S, Stephan KE (2018).
%     Variational Bayesian Inversion for Hierarchical Unsupervised
%     Generative Embedding (HUGE). NeuroImage, 179: 604-619
% 
% https://doi.org/10.1016/j.neuroimage.2018.06.073
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
function [  ] = tapas_huge_generate_examples( tname )
%% generate data from a two-region linear DCM
optionsGen = struct();
optionsGen.snr = 1; % signal-to-noise-ratio
optionsGen.N_k = [10 10]; % number of subjects per cluster
optionsGen.R = 2; % number of regions

% cluster
optionsGen.mu_k.idx = [1,3,4,5,8];
optionsGen.mu_k.value = [...
    -0.4, 0.5,-0.6, 0.3 , 0.8;...
    -0.6,-0.2,-0.4, 0.8 , 0.3];
optionsGen.sigma_k = 0.1;

% hemodynamics
optionsGen.mu_h = zeros(1,optionsGen.R*2+1);
optionsGen.sigma_h = 0.01;

% hemodynamic parameters
optionsGen.hemParam.listHem = [0.6400 2 1];
optionsGen.hemParam.scaleC = 16;
optionsGen.hemParam.echoTime = 0.0400;
optionsGen.hemParam.restingVenousVolume = 4;
optionsGen.hemParam.relaxationRateSlope = 25;
optionsGen.hemParam.frequencyOffset = 40.3000;
optionsGen.hemParam.oxygenExtractionFraction = 0.4000;
optionsGen.hemParam.rho = 4.3000;
optionsGen.hemParam.gamma = 0.3200;
optionsGen.hemParam.alphainv = 3.1250;
optionsGen.hemParam.oxygenExtractionFraction2 = 0.3200;

% input
nU = 6000;
idx1 = randi( nU - 400, 1, 75);
idx1 = bsxfun(@plus, idx1 + 200, (0:3)');
optionsGen.input.u = zeros(nU, 2);
optionsGen.input.u(idx1(:), 1) = 1;
idx2 = bsxfun(@plus, 300:600:nU-301, (1:300)');
optionsGen.input.u(idx2(:), 2) = 1;

optionsGen.input.trSteps = 20;
optionsGen.input.trSeconds = 2;

DcmInfo = tapas_huge_simulate(optionsGen);
DCMr2l = tapas_huge_export_spm(DcmInfo); %#ok<NASGU>


%% generate data from a three-region bilinear DCM
optionsGen = struct();
optionsGen.snr = 1; % signal-to-noise-ratio
optionsGen.N_k = [40 20 20]; % number of subjects per cluster
optionsGen.R = 3; % number of regions

% cluster
optionsGen.mu_k.idx = [1,2,3,4,5,6,9,10,27];
optionsGen.mu_k.value = [...
    -0.7, 0.2,-0.1,-0.2 ,-0.6, 0.3,-0.4,0.3, 0.3;...
    -0.7, 0.1, 0.3, 0.1 ,-0.4,-0.1,-0.6,0.6, 0.1;...
    -0.7,-0.2, 0.3, 0.25,-0.4,-0.1,-0.6,0.6,-0.2];
optionsGen.sigma_k = 0.1;

% hemodynamics
optionsGen.mu_h = zeros(1,optionsGen.R*2+1);
optionsGen.sigma_h = 0.01;

% hemodynamic parameters
optionsGen.hemParam.listHem = [0.6400 2 1];
optionsGen.hemParam.scaleC = 16;
optionsGen.hemParam.echoTime = 0.0400;
optionsGen.hemParam.restingVenousVolume = 4;
optionsGen.hemParam.relaxationRateSlope = 25;
optionsGen.hemParam.frequencyOffset = 40.3000;
optionsGen.hemParam.oxygenExtractionFraction = 0.4000;
optionsGen.hemParam.rho = 4.3000;
optionsGen.hemParam.gamma = 0.3200;
optionsGen.hemParam.alphainv = 3.1250;
optionsGen.hemParam.oxygenExtractionFraction2 = 0.3200;

% input
optionsGen.input.u = double([...
    reshape(repmat((1:2^9<2^8)'&(mod(1:2^9,2^5)==0)',1,2^3),2^12,1),...
    reshape(repmat((1:2^10<2^8)',1,2^2),2^12,1)]);
optionsGen.input.u = circshift(optionsGen.input.u,117,1);           
optionsGen.input.trSteps = 16;
optionsGen.input.trSeconds = 2;

DcmInfo = tapas_huge_simulate(optionsGen);
DCMr3b = tapas_huge_export_spm(DcmInfo); %#ok<NASGU>


%% save DCM in SPM format
save(tname,'DCMr2l','DCMr3b');


end

