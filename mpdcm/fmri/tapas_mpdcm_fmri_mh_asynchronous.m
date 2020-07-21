function [ps, fe] = tapas_mpdcm_fmri_mh_asynchronous(dcm, pars)
%% Estimates the posterior probability of the parameters using MCMC combined
% with path sampling.
%
% Input:
% dcm       -- Struct. DCM to be estimated
% pars      -- Struct. Parameters for the mcmc. 
%               verb: Verbose option. Defs. False. 
%               mc3i: Number of times that the exchange operator is applied.
%                   Defs. 20. 
%               diagi: Number of samples before recomputing the kernel 
%                   and producing verbose output. Defs. 200.
%               rinit: Random initilization. If true, a time dependent seed 
%                   is used and samples from the prior are used for the 
%                   initilization. If false, a VB routine is used to compute
%                   an optimal initilization. Defs. False. 
%
% Output:
% ps        -- Struct. Posterior distribution.
% fe        -- Scalar. Estimated free energy.
%
% Uses a standard MCMC on a population and applies an exchange operator
% to improve the mixing.
%
%
% Real parameter evolutionary Monte Carlo with applications to Bayes Mixture
% Models, Journal of American Statistica Association, Liang & Wong 2001.
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

T = sort(pars.T);
nt = numel(T);

nburnin = pars.nburnin;
niter = pars.niter;

% Generalize with function handles

fllh = pars.fllh;
flpp = pars.flpp;
gibbs = pars.gibbs;
prepare_ptheta = pars.prepare_ptheta;

pars.arch = 'gpu';

[y, u, theta, ptheta] = tapas_mpdcm_fmri_tinput(dcm);

% Basic initialization
ptheta = prepare_ptheta(ptheta);
ptheta.T = T;
ptheta.integ = pars.integ;
ptheta.arch = pars.arch;
ptheta.dt = pars.dt;

htheta = tapas_mpdcm_fmri_htheta(ptheta);


switch pars.rinit
    case 0
        [otheta, ollh, olpp] = init_vb(y, u, theta, ptheta, T, pars);
    case 1
        [otheta, ollh, olpp] = init_rand(y, u, theta, ptheta, T, pars);
    case 2
        [otheta, ollh, olpp] = init_prior(y, u, theta, ptheta, T, pars);
end

op = tapas_mpdcm_fmri_get_parameters(otheta, ptheta);
oy = tapas_mpdcm_fmri_predict(y, u, otheta, ptheta, 1);

ps_theta = zeros(numel(op{end}), niter);
ellh = zeros(nt, niter);
elpp = zeros(nt, niter);

diagnostics = zeros(1, nt);

ok = init_kernel(op, y, ptheta);
os = zeros(numel(op{1}), nt, pars.diagi);

t = 1;

fe_seq = zeros(floor(niter / pars.diagi) - 1, 1);

% Keep samples of the log likelihood for optimizing the schedule
dllh = zeros(nt, pars.diagi);

% Change to asynchronous mode
ptheta.arch = 'gpu_asynchronous';
sdiag = zeros(1, nt);
tic
for i = 1:nburnin + niter
    % Diagnostics and kernel update
    if i > 1 && mod(i-1, pars.diagi) == 0
        try
            toc
        end

        diagnostics = diagnostics/pars.diagi;
        
        if i > nburnin + pars.diagi
            if numel(T) > 1
                fe = trapz(T, mean(ellh(:, 1:i - nburnin), 2));
                try
                    fe_seq(floor( (i - nburnin) / pars.diagi), 1) = fe;
                end
            else 
                fe = nan;
            end
        end

        if pars.verb
            fprintf(1, 'Iter %d, diagnostics:  ', i);
            fprintf(1, '%0.2f ', diagnostics);
            fprintf(1, '\n');
            fprintf(1, 'swap:\n')
            fprintf(1, '%0.2f ', sdiag * numel(T)/(pars.diagi * pars.mc3i));
            fprintf(1, '\n');
            %fprintf(1, 'llh: ')
            %fprintf(1, '%0.2f ', ollh);
            %fprintf(1, '\n');
            if i > nburnin + pars.diagi
                fprintf(1, 'Fe: %0.05f\n', fe);
            end
        end
        % Update kernel
        if i < nburnin
            ok = tapas_mpdcm_fmri_update_kernel(2, ok, os, ...
                dllh, diagnostics, ptheta);
        end
        sdiag(:) = 0;
        diagnostics(:) = 0;
        t = t + 1;
        tic
    end

    otheta = tapas_mpdcm_fmri_set_parameters(op, otheta, ptheta);

    % Sample parameters of the metropolist part and integrate the system
    np = tapas_mpdcm_fmri_sample(op, ptheta, htheta, ok);
    ntheta = tapas_mpdcm_fmri_set_parameters(np, otheta, ptheta);
    % Make a prediction. This is done asynchronously
    container = tapas_mpdcm_fmri_predict(y, u, ntheta, ptheta, 1);
    % Leave the other parts of the algorithm run

    % Make a Gibbs step on the betas
    [otheta, oy, ollh, olpp] = gibbs(...
        y, u, otheta, ptheta, pars, oy, ollh, olpp);
    % Update the variance using a pseudo gibbs step
    [otheta, oy, ollh, olpp] = tapas_mpdcm_fmri_sample_gibbs_gamma(...
        y, u, otheta, ptheta, pars, oy, ollh, olpp);

    op = tapas_mpdcm_fmri_get_parameters(otheta, ptheta);

    ntheta = tapas_mpdcm_fmri_copy_parameters(ntheta, otheta, ...
        {'lambda', 'beta'});
    np = tapas_mpdcm_fmri_get_parameters(ntheta, ptheta);

    % Pick up the signal
    ny = tapas_mpdcm_fmri_collect_simulations(container);
    tapas_mpdcm_fmri_destroy_container(container);

    % Compute the likelihood
    nllh = fllh(y, u, ntheta, ptheta, ny);
    
    nllh = sum(nllh, 1);
    nlpp = sum(flpp(y, u, ntheta, ptheta, np), 1);

    nllh(isnan(nllh)) = -inf;
        
    v = nllh.*T + nlpp - (ollh.*T + olpp);
    tv = v;
    v = rand(size(v)) < exp(bsxfun(@min, v, 0));

    ollh(v) = nllh(v);
    olpp(v) = nlpp(v);
    op(:, v) = np(:, v);
    oy(:, v) = ny(:, v);

    assert(all(-inf < ollh), 'mpdcm:fmri:ps', '-inf value in the likelihood');

    diagnostics(:) = diagnostics(:) + v(:);

    % Population swap
    tt = 1:nt;

    for l = 1:pars.mc3i 
        % Population MCMC methods for history matching and uncertainty
        % 2012
        v0 = ceil(rand()*nt);
        if v0 == nt
            v1 = v0 - 1;
        elseif v0 == 1
            v1 = 2;
        else
            if rand() > 0.5
                v1 = v0 + 1;
            else
                v1 = v0 - 1;
            end
        end
        p = (ollh(v0) - ollh(v1)) * (T(v1) - T(v0));
        if log(rand()) < p
            ollh([v0, v1]) = ollh([v1, v0]);
            olpp([v0, v1]) = olpp([v1, v0]);
            tt([v0, v1]) = tt([v1, v0]);
            sdiag(v0) = sdiag(v0) + 1;
        end
    end
    oy(:, :) = oy(:, tt);
    op(:, :) = op(:, tt);

    % Store samples
    if i > nburnin
        ps_theta(:, i - nburnin) = op{end};
        ellh(:, i - nburnin) = ollh;
        elpp(:, i - nburnin) = olpp;
    else
        dllh(:, mod(i-1, pars.diagi) + 1) = ollh;
        os(:, :, mod(i-1, pars.diagi) + 1) = cell2mat(op);
    end
end
toc

% Collect results.

fe = trapz(T, mean(ellh, 2));

[~, i] = max(ellh(end, :) + elpp(end, :));

ps.pE = ps_theta(:, i);
ps.theta = tapas_mpdcm_fmri_set_parameters({ps.pE}, theta(1), ptheta);
ptheta.arch = 'gpu';
ps.y = tapas_mpdcm_fmri_int(u, ps.theta, ptheta);
ps.y = ps.y{:};

ps.stheta = [];
ps.sellh = [];
ps.selpp = [];
ps.ptheta = ptheta;

if pars.samples
    ps.stheta = ps_theta;
    ps.sellh = ellh;
    ps.selpp = elpp;
end
ps.F = fe;
ps.F_seq = fe_seq;
% Prior posterior mean
ps.ppm = -log(mean(exp(-ellh(end, :))));
% Prior arithmetic mean (or almost)
ps.pam = log(mean(exp(ellh(1, :))));

end

function [nk] = init_kernel(op, y, ptheta)
%% Initilize the kernel or covariance matrix of the proposal distribution.
%
% See Exploring an adaptative Metropolis Algorithm
% 

np = sum(ptheta.mhp, 1);
nr = size(y{1}, 1);
tp = np;%- nr * size(ptheta.X0, 2);

c = diag([ones(1, tp - (2 * nr + 1)), 0.0025 * ones(1, 2 * nr + 1)]); 

% Normalize the norm of the matrix
c = c./eigs(c, 1);

nk = cell(size(op, 2), 1);
nk(:) = {chol(c)};

s = linspace(0.1, 0.02, size(op, 2)).^2;

nk = struct('S', nk, 's', []);
for i = 1:size(op, 2)
    nk(i).s = s(i);
end

end


function [ntheta, nllh, nlpp] = init_rand(y, u, theta, ptheta, T, pars)
%% Randomly start each chain by sampling from the prior distribution.

mu = ptheta.p.theta.mu;

nc = numel(T);
np = numel(mu);

% Set up the random generator

rng('shuffle');

% Cholevsky decomposition of the prior precision

cpi = chol(ptheta.p.theta.pi);
mu = ptheta.p.theta.mu;
mu = repmat(mu, 1, nc);

noise = randn(np, nc);

smu = mu + 0.1 * cpi\noise;
smu = mat2cell(full(mu), [np], ones(nc, 1));

ntheta = cell(1, nc);
ntheta(:) = theta(:);

ntheta = tapas_mpdcm_fmri_set_parameters(smu, ntheta, ptheta);
ny = tapas_mpdcm_fmri_predict(y, u, ntheta, ptheta);
nllh = pars.fllh(y, u, ntheta, ptheta, ny);
nlpp = pars.flpp(y, u, ntheta, ptheta);

end

function [otheta, ollh, olpp] = init_vb(y, u, theta, ptheta, T, pars)
% Create an initial estimate to reduce burn in phase by computing the 
% posterior distribution of the power posteriors

nt = numel(T);
op = cell(1, nt);
otheta = theta;
for i = nt:-1:1
    ptheta.T = T(i);
    try
        [q, otheta] = tapas_mpdcm_fmri_gmodel(y, u, otheta, ptheta);
        ny = tapas_mpdcm_fmri_predict(y, u, otheta, ptheta);   
        ollh = pars.fllh(y, u, otheta, ptheta, ny);
        olpp = pars.flpp(y, u, otheta, ptheta);
        if pars.verb
            fprintf(1, 'Starting llh: %0.5f\n', ollh + olpp);
        end
        op(i) = tapas_mpdcm_fmri_get_parameters(otheta, ptheta);
    catch err
        if strcmp(err.identifier, 'tapas:mpdcm:fmri:mle:numeric') || ...
            numel(err.identifier) == 0
            op{i} = full(ptheta.p.theta.mu);
            otheta = tapas_mpdcm_fmri_set_parameters(op(i), theta, ptheta); 
        else
            rethrow(err)
        end
    end
end

% Samples from posterior

theta = cell(1, nt);
theta(:) = otheta;

% Fully initialize

otheta = tapas_mpdcm_fmri_set_parameters(op, theta, ptheta);

ny = tapas_mpdcm_fmri_predict(y, u, otheta, ptheta);
ollh = pars.fllh(y, u, otheta, ptheta, ny);
olpp = pars.flpp(y, u, otheta, ptheta);

% Eliminate weird cases

[~, l] = min(abs(bsxfun(@minus, find(isnan(ollh)), find(~isnan(ollh))')));
tl = find(~isnan(ollh));
l = tl(l);

olpp(isnan(ollh)) = olpp(l);
op(:, isnan(ollh)) = op(:, l);
ollh(isnan(ollh)) = ollh(l);

end

function [otheta, ollh, olpp] = init_prior(y, u, theta, ptheta, T, pars);
%% Initlize from the prior

nt = numel(T);
    
otheta = cell(1, nt);
ttheta = theta{1};
ttheta.A(:) = 0;
ttheta.B(:) = 0;
ttheta.C(:) = 0;
ttheta.D(:) = 0;

ttheta = {ttheta};

otheta(:) = ttheta(:);
oy = tapas_mpdcm_fmri_predict(y, u, otheta, ptheta);
ollh = pars.fllh(y, u, otheta, ptheta, oy);

ollh = sum(ollh, 1);
olpp = sum(pars.flpp(y, u, otheta, ptheta), 1);

end

