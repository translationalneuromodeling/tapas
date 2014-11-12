function [y, u, theta, ptheta] = dcm_fmri_tinput(dcm)
%% Transforms input that follows the SPM API into valid mpdcm input.
%
% Input:
%
% dcm -- Cell array of SPM DCM models.
%
% Output:
%
% y -- Data array
% u -- Input array
% theta -- Parameters structure
% ptheta -- Hyperparameters
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

ns = size(dcm.Y.y, 1);
nr = size(dcm.Y.y, 2);

% Data

y = dcm.Y.y';


% Priors

ptheta = struct('dt', 1.0, 'dyu', []);

[pE, pC, x] = spm_dcm_fmri_priors(dcm.a, dcm.b, dcm.c);

% hyperpriors - Basis matrixes

try
    ptheta.Q = dcm.Y.Q;
catch err
    if strcmp(err.identifier, 'MATLAB:nonExistentField');
        ptheta.Q = spm_Ce(ns*ones(1, nr));
    end
end

nh = numel(ptheta.Q);

nd = size(ptheta.Q{1}, 1);

tQ = speye(nd);
for i = 1:nh
    tQ = tQ + sparse(ptheta.Q{i});
end

dm = all(find(tQ)' == (nd*(0:nd-1) + (1:nd)));

if dm
    ptheta.dQ.dm = 1;
    ptheta.dQ.Q = cell(1, nh);
    for i = 1:nh
        ptheta.dQ.Q{i} = diag(ptheta.Q{i});
    end
else
    ptheta.dQ.dm = 0;
    ptheta.dQ.Q = {};
end


try
    hE = dcm.M.hE(:);
catch err
    if strcmp(err.identifier, 'MATLAB:nonExistentField');
        hE = sparse(nh,1);
    end
end

% hyperpriors - covariance
try
    hC = dcm.M.hC;
catch err
    if strcmp(err.identifier, 'MATLAB:nonExistentField');
        hC = speye(nh,nh);
    end
end


v = [ logical(dcm.a(:)); logical(dcm.b(:)); logical(dcm.c(:)); ...
    ones(nr + nr + 1 + nh, 1)];
v = logical(v);

ptheta.mtheta = [pE.A(:); pE.B(:); ...
    pE.C(:); pE.transit(:); pE.decay(:); ...
    pE.epsilon(:); hE(:)];

ptheta.mtheta = ptheta.mtheta(v);

ptheta.ctheta = sparse(blkdiag(pC, hC));
ptheta.ctheta = ptheta.ctheta(v, v);
ptheta.ictheta = inv(ptheta.ctheta);
ptheta.chol_ictheta = chol(ptheta.ictheta);

ptheta.a = dcm.a;
ptheta.b = dcm.b;
ptheta.c = dcm.c;


% Parametrization from spm8

theta = struct('A', [], 'B', [], 'C', [], 'epsilon', [], 'K', [], ...
    'tau',  [], 'V0', [], 'E0', [], 'k1', [], 'k2', [], 'k3', [], ...
    'alpha', [], 'gamma', [], 'dim_x', [], 'dim_u', [], ...
    'fA', 1, 'fB', 1, 'fC', 1 );

decay = 0;
transit = 0;

gamma   = 0.32;
alpha   = 0.32;
E0      = 0.32;
V0      = 4.0;

ep = exp(full(pE.epsilon));

tau     = 2*exp(transit);
kappa   = 0.64*exp(decay);

theta.dim_x = size(dcm.a, 1);
theta.dim_u = size(dcm.c, 2);

theta.A = full(pE.A);
if any(dcm.a(:))
    theta.fA = 1;
else
    theta.fA = 0;
end

theta.B = cell(theta.dim_u, 1);

for j = 1:theta.dim_u
    theta.B{j} = full(pE.B(:,:,j));
end

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

theta.epsilon = ep;
theta.K = kappa.*ones(theta.dim_x, 1);
theta.tau = tau.*ones(theta.dim_x, 1);
theta.E0 = E0;
theta.V0 = V0;
theta.alpha = alpha;
theta.gamma = gamma;

[k1, k2, k3] = mpdcm_fmri_k(theta, ptheta);

theta.k1 = k1;
theta.k2 = k2;
theta.k3 = k3;

% Noise

theta.lambda = hE;

dyu = dcm.U.dt;

u = dcm.U.u';

% Sampling frequency

ptheta.dyu = 2.0*size(y, 2)/size(u, 2);

end

