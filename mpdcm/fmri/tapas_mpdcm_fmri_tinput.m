function [y, u, theta, ptheta] = tapas_mpdcm_fmri_tinput(dcm)
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
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%

nm = numel(dcm);

y = cell(nm, 1);
u = cell(nm, 1);
theta = cell(nm, 1);

[ys, us] = max_size(dcm);

scale = zeros(nm, 1);
dyu = zeros(nm, 1);
udt = zeros(nm, 1);

for i = 1:nm
    [y{i} scale(i)] = tinput_y(dcm{i}, max(ys));
    [u{i} dyu(i) udt(i)] = tinput_u(dcm{i}, max(us));
    theta{i} = tinput_theta(dcm{i});
end

ptheta = tinput_ptheta(dcm, scale, dyu, udt, ys, us);

end

%% ===========================================================================

function [ys, us] = max_size(dcm)
    % Obtain the max lenght of the arrays

    ys = zeros(numel(dcm), 1);
    us = zeros(numel(dcm), 1);

    for i = 1:numel(dcm)
        ys(i) = size(dcm{i}.Y.y, 1);
        us(i) = size(dcm{i}.U.u, 1);
    end

end

%% ===========================================================================

function [ptheta] = tinput_ptheta(dcm, scale, dyu, udt, ys, us)
    % Priors

    nr = size(dcm{1}.Y.y, 2);
   
    if isfield(dcm{1}.Y, 'X0')
        % Size of the basis of the null space
        nb = size(dcm{1}.Y.X0, 2);
    else
        nb = 0;
    end
    ptheta = struct('dt', 1.0, 'udt', [], 'dyu', [], 'rescale', scale, ...
        'ys', ys, 'us', us);

    Q = dcm{1}.Y.Q;
    M = dcm{1}.M;
    a = dcm{1}.a;
    b = dcm{1}.b;
    c = dcm{1}.c;
    d = dcm{1}.d;

    if ~any(d)
        d = zeros(size(a, 1), size(a, 1), size(a, 1));
    end

    [pE, pC, x] = spm_dcm_fmri_priors(a, b, c, d);

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

    % Prior of the betas.
    
    ptheta.X0_variance = 16;

    pB = ptheta.X0_variance * eye(nr * nb);

    v = [ logical(a(:)); logical(b(:)); logical(c(:)); logical(d(:)); 
        ones(nr + nr + 1 + nh, 1)];
    v = logical(v);

    % Lambdas are included in the same parameters.
    mtheta = [pE.A(:); pE.B(:); ...
        pE.C(:); pE.D(:); pE.transit(:); pE.decay(:); ...
        pE.epsilon(:); hE(:)];

    ctheta = sparse(blkdiag(pC, hC));
    ctheta = ctheta(v, v);
    ctheta = sparse(blkdiag(full(ctheta), pB));

    ptheta.p.theta.mu = [mtheta(v); zeros(nr * nb, 1)];
    ptheta.p.theta.pi = inv(ctheta);
    ptheta.p.theta.sigma = ctheta;

    ptheta.a = logical(a);
    ptheta.b = logical(b);
    ptheta.c = logical(c);
    ptheta.d = logical(d);

    ptheta.i_a = find(a);
    ptheta.i_b = find(b);
    ptheta.i_c = find(c);
    ptheta.i_d = find(d);

    ptheta.n_a = sum(ptheta.a(:));
    ptheta.n_b = sum(ptheta.b(:));
    ptheta.n_c = sum(ptheta.c(:));
    ptheta.n_d = sum(ptheta.d(:));

    assert(all(dyu == dyu(1)), 'mpdcm:fmri:tinput:dyu', ...
        'Sampling rates not equal across data sets');
    assert(all(udt == udt(1)), 'mpdcm:fmri:tinput:dyu', ...
        'U.dt is not equal across data sets');

    ptheta.dyu = dyu(1);
    ptheta.udt = udt(1);

    ptheta.X0 = [];
    if isfield(dcm{1}.Y, 'X0')
        ptheta.X0 = dcm{1}.Y.X0;
    end

    % Parameters samples using metrolis hastings
    ptheta.mhp = logical([ones(sum(v) - nr, 1); 0 * ones(nr, 1); ...
        0 * ones(nr * nb, 1)]);
end

function [u, dyu, udt] = tinput_u(dcm, us)
%% Prepare U

    U = dcm.U;
    [u, udt] = tapas_mpdcm_fmri_init_u(dcm.U, dcm);

    % Sampling frequency
    dyu = size(dcm.Y.y, 1)/size(U.u, 1);

end

%% ===========================================================================

function [y, scale] = tinput_y(dcm, ys)
    % Prepare y

    ns = size(dcm.Y.y, 1);
    nr = size(dcm.Y.y, 2);

    % Data

    Y = dcm.Y;

   
    try
        if dcm.options.detrend_y
            fprintf(1, 'Detrend Y.y\n');
            Y.y = spm_detrend(Y.y);
        end
    end

    scale = 1;
    try
    if dcm.options.rescale_y
        fprintf(1, 'Rescale input Y.y\n')
        if isfield(Y, 'scale')
            scale = Y.scale;
        else
            scale   = max(max((Y.y))) - min(min((Y.y)));
            scale   = 4/max(scale,4);
        end
    end
    end
    Y.y = Y.y * scale;
    y   = Y.y';

end

%% ===========================================================================

function [theta] = tinput_theta(dcm)
    % A single theta set of parameters

    [pE, ~, ~] = spm_dcm_fmri_priors(dcm.a, dcm.b, dcm.c, dcm.d);

    theta = tapas_mpdcm_fmri_init_theta(pE, dcm);

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
