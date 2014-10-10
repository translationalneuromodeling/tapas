function [u, theta, ptheta] = dcm_fmri_tinput(U, P)
%% Creates input that follows the SPM API into valid input.
%
% M -- Cell array of SPM DCM models.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

theta0 = struct('A', [], 'B', [], 'C', [], ...
    'epsilon', [], 'K', [], 'tau',  [], ...
    'V0', [], 'E0', [], 'k1', [], 'k2', [], 'k3', [], ... 
    'alpha', [], 'gamma', [], 'dim_x', [], 'dim_u', [], ...
    'fA', 1, 'fB', 1, 'fC', 1 );

ptheta = struct('dt', 0.5, 'dyu', []);

% Parametrization from spm8

decay = 0;
transit = 0;

gamma   = 0.32;
alpha   = 0.32;
E0      = 0.32;
V0      = 4.0;
r0      = 25;
nu0     = 40.3;
TE      = 0.04;


theta   = cell(size(P));
u       = cell(size(U));

for i = 1:numel(P)

    ep = exp(full(P{i}.epsilon));

    tau     = 2*exp(transit);
    kappa   = 0.64*exp(decay);

    k1      = 4.3*nu0*0.4*TE;
    k2      = ep*r0*0.4*TE;
    k3      = 1 - ep;

    theta0.dim_x = size(P{i}.A, 1);
    theta0.dim_u = size(P{i}.C, 2);

    theta0.A = full(P{i}.A);
    if any(theta0.A)
        theta0.fA = 0;
    else 
        theta0.fA = 1;
    end

    theta0.B = cell(theta0.dim_u, 1);

    for j = 1:theta0.dim_u
        theta0.B{j} = full(P{i}.B(:,:,j));
    end

    if any(P{i}.B)
        theta0.fB = 0;
    else
        theta0.fB = 1;
    end

    theta0.C = full(P{i}.C)/16;
    
    if any(theta0.C)
        theta0.fC = 1;
    else
        theta0.fC = 0;
    end

    theta0.epsilon = ep*ones(theta0.dim_x, 1);
    theta0.K = kappa*ones(theta0.dim_x, 1);
    theta0.tau = tau*ones(theta0.dim_x, 1);
    theta0.E0 = E0;
    theta0.V0 = V0;
    theta0.alpha = alpha;
    theta0.gamma = gamma;
    theta0.k1 = k1;
    theta0.k2 = k2;
    theta0.k3 = k3;

    theta{i} = theta0;

end

u = cell(size(u));

dyu = U{1}.dt;
dp = 0;
for i = 1:numel(U)
    if size(U{1}.u, 1) > dp
        dp = size(U{i}.u, 1);
    end
end

for i = 1:numel(U)
    assert(dyu == U{i}.dt, 'dcm:fmri:dcm_fmri_tinput:U:dyu:not_match', ...
        'Rates of sampling rate don''t match.')
    u0 = U{i}.u'; 
    nu = nan(size(u0, 1), dp);
    nu(1:size(u0, 1), 1:size(u0, 2)) = u0;
    u{i} = nu;
end 

ptheta.dyu = 0.5*dyu;

end
