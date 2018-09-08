function [theta] = tapas_mpdcm_fmri_init_theta(pE, dcm)
%% Initilizes theta for the generation of data. 
%
% Input
%   pE          Parameter structure.
%   dcm         Structure with fields a, b, c, and d denoting degrees of 
%                   in each of the matrices.
%   
% Output
%   theta       Structure compatible with the integrator.
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%

n = 2;
if nargin < n
    dcm.a = logical(pE.A);
    dcm.b = logical(pE.B);
    dcm.c = logical(pE.C);
    dcm.d = logical(pE.D);
end

theta = struct('A', [], 'B', [], 'C', [], 'D', [], ...
    'TE', [], 'epsilon', [], 'K', [], ...
    'tau',  [], 'V0', [], 'E0', [], 'k1', [], 'k2', [], 'k3', [], ...
    'alpha', [], 'gamma', [], 'dim_x', [], 'dim_u', [], 'ny', [], ...
    'fA', 1, 'fB', 1, 'fC', 1 , 'fD', 0);


hps = tapas_mpdcm_fmri_get_hempars();

gamma   = hps.gamma;
alpha   = hps.alpha;
E0      = hps.E0;
V0      = hps.V0;

ep = full(pE.epsilon);
transit = full(pE.transit);
decay = full(pE.decay);

tau = transit;
K = decay;

theta.dim_x = size(dcm.a, 1);
theta.dim_u = size(dcm.c, 2);

theta.A = full(pE.A);
if any(dcm.a(:))
    theta.fA = 1;
else
    theta.fA = 0;
end

theta.B = full(pE.B);
if any(dcm.b(:))
    theta.fB = 1;
else
    theta.fB = 0;
end

theta.C = full(pE.C);
if any(dcm.c(:))
    theta.fC = 1;
else
    theta.fC = 0;
end

theta.D = full(pE.D);
if any(dcm.d(:))
    theta.fD = 1;
else
    theta.fD = 0;
end

if size(theta.D, 3) == 0
    theta.D = zeros(theta.dim_x, theta.dim_x, theta.dim_x);
end

if isfield(dcm, 'TE')
    theta.TE = dcm.TE;
elseif isfield(pE, 'TE')
    theta.TE = pE.TE;
else
    theta.TE = 0.04;
end

theta.epsilon = ep;
theta.K = K .* ones(theta.dim_x, 1);
theta.tau = tau .* ones(theta.dim_x, 1);
theta.E0 = E0;
theta.V0 = V0;
theta.alpha = alpha;
theta.gamma = gamma;

if isfield(dcm, 'Y')
    if isfield(dcm.Y, 'X0')
        theta.beta = zeros(size(dcm.Y.X0, 2), theta.dim_x);
    end
end
end % tapas_mpdcm_fmri_init_theta 

