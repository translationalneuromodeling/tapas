function [ DCM ] = tapas_ceode_testData_cmc()
% [ DCM ] = tapas_ceode_testData_cmc()
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
M.x = zeros(2, 8);
M.m = 1;


% ---------------------- DESIGN SPECIFICATION -----------------------------
xU = struct();

% No condition specific effec
xU.X = [0]';
xU.name = {'std'};

% Sampling rate of the artificial signal
xU.dt = 1e-03;


% ---------------------- PARAMETER VALUES ---------------------------------
% Neuronal parameters (also see spm_fx_cmc.m / tapas_ceode_fx_cmc.m)
Ep = struct();

Ep.M = zeros(2, 2);
Ep.T = [0 0 0 0]; 
Ep.G = zeros(2, 3);
Ep.A{1} = [-32, -32; 0, -32];
Ep.A{2} = [-32, -32; 0, -32]; 
Ep.A{3} = [-32, 0; -32, -32];
Ep.A{4} = [-32, 0; -32, -32];
Ep.B{1} = [0 0; 1 0];
Ep.N{1} = zeros(2, 2);
Ep.C = [0; -32];
Ep.D = -32 * ones(2, 2);
Ep.S =  0; 
Ep.R = [0 0]; 

% Leadfield parameters (also see spm_lx_erp.m)
Eg.L = [-1 0];
Eg.J = [0 0 4 0 0 0 2 0];


% ---------------------- CREATE OUTPUT ------------------------------------
% Combine all structures into single DCM structure
DCM.Ep = Ep;
DCM.Eg = Eg;
DCM.M = M;
DCM.xU = xU;


end

