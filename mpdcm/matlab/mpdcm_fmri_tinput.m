function [y, u, theta, ptheta] = mpdcm_fmri_tinput(dcm)
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

nm = numel(dcm);
ns = size(dcm{1}.Y.y, 1);
nr = size(dcm{1}.Y.y, 2);

y = cell(nm, 1);
u = cell(nm, 1);
theta = cell(nm, 1);

scale = zeros(nm, 1);
dyu = zeros(nm, 1);

for i = 1:nm
    [y{i} scale(i)] = tinput_y(dcm{i});
    [u{i} dyu(i)] = tinput_u(dcm{i});
    theta{i} = tinput_theta(dcm{i});
end

assert(all(dyu == dyu(1)), 'mpdcm:fmri:tinput:dyu', ...
    'Ratio of sampling rate between y and u are not equal across data sets');

% Priors

ptheta = struct('dt', 1.0, 'dyu', [], 'rescale', scale);

Q = dcm{1}.Y.Q;
M = dcm{1}.M;
a = dcm{1}.a;
b = dcm{1}.b;
c = dcm{1}.c;

[pE, pC, x] = spm_dcm_fmri_priors(a, b, c);

% hyperpriors - Basis matrixes

try
    ptheta.Q = Q;
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

% Basis of the covariances

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

% Hyperprios expected value
try
    hE = M.hE(:);
catch err
    if strcmp(err.identifier, 'MATLAB:nonExistentField');
        hE = sparse(nh,1);
    end
end


try
    hC = M.hC;
catch err
    if strcmp(err.identifier, 'MATLAB:nonExistentField');
        hC = speye(nh,nh);
    end
end


v = [ logical(a(:)); logical(b(:)); logical(c(:)); 
    ones(nr + nr + 1 + nh, 1)];
v = logical(v);

mtheta = [pE.A(:); pE.B(:); ...
    pE.C(:); pE.transit(:); pE.decay(:); ...
    pE.epsilon(:); hE(:)];

ctheta = sparse(blkdiag(pC, hC));
ctheta = ctheta(v, v);

ptheta.p.theta.mu = mtheta(v);
ptheta.p.theta.pi = inv(ctheta);
ptheta.p.theta.chol_pi = chol(ptheta.p.theta.pi);
ptheta.p.theta.sigma = ctheta;

ptheta.a = a;
ptheta.b = b;
ptheta.c = c;

ptheta.dyu = dyu(1);

end

%% ===========================================================================


function [u, dyu] = tinput_u(dcm)
%% Prepare U

    y = dcm.Y.y;

    dyu = dcm.U.dt;

    u = dcm.U.u';

    % Sampling frequency

    dyu = 2.0*size(y, 1)/size(u, 2);

end


function [y, scale] = tinput_y(dcm)
% Prepare y

    ns = size(dcm.Y.y, 1);
    nr = size(dcm.Y.y, 2);

    % Data

    Y = dcm.Y;

    scale   = max(max((Y.y))) - min(min((Y.y)));
    scale   = 4/max(scale,4);
    Y.y     = Y.y*scale;

    y = Y.y';

end

%% ===========================================================================

function [theta] = tinput_theta(dcm)
    % A single theta set of parameters

    [pE, pC, x] = spm_dcm_fmri_priors(dcm.a, dcm.b, dcm.c);

    theta = struct('A', [], 'B', [], 'C', [], 'epsilon', [], 'K', [], ...
        'tau',  [], 'V0', [], 'E0', [], 'k1', [], 'k2', [], 'k3', [], ...
        'alpha', [], 'gamma', [], 'dim_x', [], 'dim_u', [], ...
        'fA', 1, 'fB', 1, 'fC', 1 );

    decay = 0;
    transit = 0;

    hps = mpdcm_fmri_get_hempars();

    gamma   = hps.gamma;
    alpha   = hps.alpha;
    E0      = hps.E0;
    V0      = hps.V0;

    ep = exp(full(pE.epsilon));

    tau     = transit;
    kappa   = decay;

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

    [k1, k2, k3] = mpdcm_fmri_k(theta);

    theta.k1 = k1;
    theta.k2 = k2;
    theta.k3 = k3;
    
    nh = numel(dcm.Y.Q);

    % Hyperprios expected value
    try
        hE = dcm.M.hE(:);
    catch err
        if strcmp(err.identifier, 'MATLAB:nonExistentField');
            hE = sparse(nh,1);
        end
    end

    theta.lambda = hE;

end
