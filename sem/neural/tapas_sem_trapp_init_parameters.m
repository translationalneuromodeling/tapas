function [theta, ptheta] = tapas_sem_trapp_init_parameters(n_nodes)
%% Generate the matrices neccessary for the model.
%
% Input
%   n_nodes       -- Number of nodes
%
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2017
%

DT = 1.0; % One millisecond 

% Input
T_FIX_OFF = 200;
T_EX_ON = 500;
T_EX_OFF = 900;

% Delays ms
T_EX_DELAY = 70; 
T_EN_DELAY = 120;

% Time constant
TAU_EX_ON = 10;
TAU_EX_OFF = 70;

SIGMA_IN = 0.7; % mm
TAU = 10; % ms
BETA = 0.07; % Slope
THETA = 0; % baseline
IMA = 144; % Interaction matrix
IMB = 48; %
IMBC = 16;
SIGMA_A = 0.6; % mm
SIGMA_B = 3 * SIGMA_A; % mm
SIGMA_ETA = 1; % Variance of the noise

% In mm in SP 
LOC_EN = 0;
LOC_EX = 0;

A_ENDO = 10;
A_EXO = 50;

dist = linspace(-5, 5, n_nodes)'; % mm
kernel = dist * ones(1, n_nodes) - ones(n_nodes, 1) * dist';
kernel = kernel .* kernel;

% Self decay
A = -eye(n_nodes);

% Matrix for connectivity 
B = 0.01 * IMA * exp(-kernel / ( 2 * SIGMA_A * SIGMA_A)) - ...
    IMB * exp(-kernel/(2 * SIGMA_B * SIGMA_B)) - IMBC;

% Matrix for the input
C = eye(n_nodes);
C(:, n_nodes + 1) = 0;
C(1:2:end, n_nodes + 1) = -100;

x0 = zeros(n_nodes, 1);

beta = 0.07 * ones(n_nodes, 1);
theta = THETA * ones(n_nodes, 1);

theta = struct(...
    'A', A, ...
    'B', B, ...
    'C', C, ...
    'beta', beta, ...
    'x0', x0, ...
    'tau', TAU, ...
    'theta', theta, ...
    'sigma_in', SIGMA_IN, ...
    'sepsilon', SIGMA_ETA, ...
    'dist', dist, ...
    'a_en', A_ENDO, ...
    'a_ex', A_EXO, ...
    'n_nodes', n_nodes, ...
    'dim_x', n_nodes, ...
    'dim_u', n_nodes + 1, ...
    't_fix_off', T_FIX_OFF, ...
    't_ex_on', T_EX_ON, ...
    't_ex_off', T_EX_OFF, ...
    't_ex_delay', T_EX_DELAY, ...
    't_en_delay', T_EN_DELAY, ...
    'tau_ex_on', TAU_EX_ON, ...
    'tau_ex_off', TAU_EX_OFF, ...
    'loc_en', LOC_EN, ...
    'loc_ex', LOC_EX);

ptheta = struct( ...
    'dt', DT, ...
    'dyu', 1, ...
    'udt', 1);


end

