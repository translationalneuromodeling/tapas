function [ DCM ] = tapas_ceode_testData_erp()
% [ DCM ] = tapas_ceode_testData_erp()
%
% THIS IS A SCRIPT SOLELY FOR TESTING THE SPM/TAPAS SETUP. DO NOT CHANGE!
%
% Sets parameters and function handles to create a synthetic DCM
% structure. The resulting structure can then be used to generate
% artificial data with tapas_ceode_gen_erp.m
%
% INPUT
%
% OUTPUT
%   DCM         struct          DCM structure with all necessary fields to
%                               generate synthetic data.
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


% ---------------------- MODEL SPECIFICATION ------------------------------
M = struct();

% Handles to specify integration scheme, dynamical equation, forward model
% and input
M.IS = 'tapas_ceode_gen_erp';
M.f = 'spm_fx_cmc';
M.int = 'spm_int_L';
M.G = 'spm_lx_erp';
M.FS = 'spm_fy_erp';
M.fu = 'spm_erp_u';

% Design/Parameter values of the driving input (see spm_erp_u.m)
M.dur = 16;
M.ons = 64;
M.ns = 500;
M.x = zeros(2, 9); 


% ---------------------- DESIGN SPECIFICATION -----------------------------
xU = struct();

% No condition specific effec
xU.X = [0]';
xU.name = {'std'};

% Sampling rate of the artificial signal
xU.dt = 1e-03;


% ---------------------- PARAMETER VALUES ---------------------------------
% Neuronal parameters (also see spm_fx_erp.m / tapas_ceode_fx_erp.m)
Ep = struct();

Ep.T = [0, 0; 0 0];  
Ep.G = [0, 0; 0 0];
Ep.S = [0, 0];
Ep.A{1} = [-4, -4; 1, -4];
Ep.A{2} = [-4, 0; -4, -4]; 
Ep.A{3} = -4 * ones(2, 2);
Ep.B{1} = zeros(2, 2);
Ep.C = [0; -4]; 
Ep.H = [0 0 0 0]; 
Ep.D = -10 * ones(2, 2);
Ep.R = [0 0]; 

% Leadfield parameters (also see spm_lx_erp.m)
Eg.L = [1, 1];
Eg.J = [1, zeros(1, 5), 1, 0, 1];


% ---------------------- CREATE OUTPUT ------------------------------------
% Combine all structures into single DCM structure
DCM.Ep = Ep;
DCM.Eg = Eg;
DCM.M = M;
DCM.xU = xU;



end

