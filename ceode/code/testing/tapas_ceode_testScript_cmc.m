function [ testResult ] = tapas_ceode_testScript_cmc()
% [  ] = tapas_ceode_testScript_cmc()
%
% THIS IS A SCRIPT SOLELY FOR TESTING THE SPM/TAPAS SETUP. DO NOT CHANGE!
%
% Takes a synthetic dataset (3 population ERP model), and computes the 
% predicted response for different integrators and levels of delays. The 
% continuous extension for ODE method (ceode)is taken as reference.
%
% INPUT
%
% OUTPUT
%   testResult        struct        Result of the setup-check             
%
% -------------------------------------------------------------------------
%
% Author: Dario Schöbi
% Created: 2020-08-10
% Copyright (C) 2020 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS ceode Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------


% Load DCM template
DCM = tapas_ceode_testData_cmc();

% Integrators
intSpec  = {...
        'spm_int_L', 'spm_fx_cmc'; ...
    'tapas_ceode_int_euler', 'tapas_ceode_fx_cmc'};
nIntegrators = size(intSpec, 1);

% Sets of Delays
tau = linspace(-1, -0.5, 5);

% Threshold for testing the setup (in terms of a correlation coefficient)
BENCHMARK = [0.9932, 0.9785, 0.9730, 0.8489, 0.4162];
ERRORMARGIN = 1E-2;

% Pyramidal cell voltage states
vPyramids = [7, 8];

% Preallocate signals
x = cell(length(intSpec), 1);
for i = 1 : length(x)
    x{i} = zeros(length(tau), length(vPyramids) * DCM.M.ns);
end


%------------------------- GENERATE PREDICTIONS ---------------------------

% Integrate the DDE with the different integrators
tau_idx = 1;
for delays = tau
    
    % Add delay to forward connection (read as FROM (columns) TO (rows))
    DCM.Ep.D(2, 1) = delays;
    
    % Iterate over integrator specs in DCM structure and generate signal
    for i = 1 : size(intSpec, 1)
        DCM.M.int = intSpec{i, 1};
        DCM.M.f = intSpec{i, 2};
        
        y = tapas_ceode_gen_erp(DCM.Ep, DCM.M, DCM.xU);
        x{i}(tau_idx, :) = spm_vec(y{1}(:, vPyramids));
    end

    % Increase index for storing signal in single structure
    tau_idx = tau_idx + 1;
end


%------------------------------ TESTING ----------------------------------

% Time samples to compute pearson correlation (only the signal from the
% second (delayed region is used))
rsamples = DCM.M.ns + 1 : 2 * DCM.M.ns; 

for i = 1 : nIntegrators-1
    
    % X-Axis offset for dot-plot
    xoffs = linspace(-0.2, 0.2, length(tau));
    
    for j = 1 : length(tau)
        rho = corrcoef(x{i}(j, rsamples), ...
            x{nIntegrators}(j, rsamples));
        rCoeff(i, j) = rho(1, 2);
    end
    
end

% Check regression coefficients against threshold
testResult = struct();

if any(abs(rCoeff - BENCHMARK) > ERRORMARGIN)
    testResult.score = 0;
    testResult.msg = ...
        sprintf(['There is a larger than expected error in the CMC model. \n ' ...
    'Compatiblity with your SPM setup can not be guaranteed. ' ...
    'Please check the README.md.']);
else
    testResult.score = 1;
    testResult.msg = ['The CMC test was successful'];
end


end








