function [ps, fe] = tapas_mpdcm_fmri_hmm(dcm, pars)
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
nburnin = pars.nburnin;
niter = pars.niter;

% Generalize with function handles

prepare_ptheta = pars.prepare_ptheta;

[y, u, theta, ptheta] = tapas_mpdcm_fmri_tinput(dcm);

% Basic initialization
ptheta = prepare_ptheta(ptheta);
ptheta.integ = pars.integ;
ptheta.arch = pars.arch;
ptheta.dt = pars.dt;
ptheta.T = T;

htheta = tapas_mpdcm_fmri_htheta(ptheta);
htheta.nms = pars.nms;
htheta.dt = linspace(0.002, 0.002, numel(T));

nt = numel(T);
fe = nan;

switch pars.rinit
case 0
    [otheta, ollh, olpp] = init_prior(y, u, theta, ptheta, T, pars);
case 1
    [otheta, ollh, olpp] = init_rand(y, u, theta, ptheta, T, pars);
case 2
    [otheta, ollh, olpp] = init_vb(y, u, theta, ptheta, T, pars);
end

op = tapas_mpdcm_fmri_get_parameters(otheta, ptheta);
oy = tapas_mpdcm_fmri_predict(y, u, otheta, ptheta);

ps_theta = zeros(numel(op{end}), niter);
ellh = zeros(nt, niter);
elpp = zeros(nt, niter);

diagnostics = zeros(1, nt);

fe_seq = zeros(floor(niter / pars.diagi) - 1, 1);
tic
for i = 1:nburnin + niter
    % Diagnostics and kernel update
    if pars.verb
        fprintf(1, 'Iteration: %d\n', i)
    end
    if i > 1 && mod(i - 1, pars.diagi) == 0
        try
            toc
        end
        diagnostics = diagnostics/pars.diagi;
        if i > nburnin + pars.diagi && nt > 1
            fe = trapz(T, mean(ellh(:, 1:i - nburnin), 2));
            try
                fe_seq(floor( (i - nburnin) / pars.diagi), 1) = fe;
            end
        end

        if pars.verb
            fprintf(1, 'Iter %d, diagnostics:  ', i);
            fprintf(1, '%0.2f ', diagnostics);
            fprintf(1, '%0.2f ', ollh + olpp);
            fprintf(1, '\n');
            if i > nburnin + pars.diagi && nt > 1
                fprintf(1, 'Fe: %0.05f\n', fe);
            end
        end
        if i < nburnin
            tdt = log2(htheta.dt) - (0.651 - diagnostics)./(2 * 0.651);
            ndt = exp(tdt * log(2));
            cv = (ndt < 0.01) .* (ndt > 0.00001);
            htheta.dt(logical(cv)) = ndt(logical(cv));
            fprintf(1, 'dt: %0.09f', htheta.dt);
            fprintf(1, '\n');
        end
        diagnostics(:) = 0;
        tic
    end

    [op, oy, ollh, olpp, v] = tapas_mpdcm_fmri_sample_hmm(y, u, ...
        otheta, ptheta, htheta, oy, op, ollh, olpp); 

    assert(all(-inf < ollh), 'mpdcm:fmri:ps', '-inf value in the likelihood');

    diagnostics(:) = diagnostics(:) + v(:);

    % Store samples

    if i > nburnin
        ps_theta(:, i - nburnin) = op{end};
        ellh(:, i - nburnin) = ollh;
        elpp(:, i - nburnin) = olpp;
    else
        os(:, :, mod(i - 1, pars.diagi) + 1) = cell2mat(op);
    end

    % Population swap

    for l = 1:pars.mc3i
        v = ceil(rand()*(nt-1));
        p = min(1, exp(ollh(v) * T(v+1) + ollh(v+1) * T(v) ...
            - ollh(v) * T(v) - ollh(v+1) * T(v+1)));
        if rand() < p
            ollh([v v+1]) = ollh([v+1 v]);
            olpp([v v+1]) = olpp([v+1 v]);
            op(:, [v v+1]) = op(:, [v+1 v]);       
        end
    end

end
toc

% Collect results.
if nt > 1
    fe = trapz(T, mean(ellh, 2));
end

ps.pE = mean(ps_theta , 2);
ps.theta = tapas_mpdcm_fmri_set_parameters({ps.pE}, theta(1), ptheta);
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

smu = mu + cpi\noise;
smu = mat2cell(full(mu), [np], ones(nc, 1));

ntheta = cell(1, nc);
ntheta(:) = theta(:);

ntheta = tapas_mpdcm_fmri_set_parameters(smu, ntheta, ptheta);
ny = tapas_mpdcm_fmri_predict(y, u, ntheta, ptheta);
nllh = pars.fllh(y, u, ntheta, ptheta, ny);
nlpp = pars.flpp(y, u, ntheta, ptheta);

end


function [otheta, ollh, olpp] = init_prior(y, u, theta, ptheta, T, pars);
%% Initlize from the prior

nt = numel(T);
    
otheta = cell(1, nt);
ttheta = theta{1};
ttheta.A(:) = 0;
ttheta.B(:) = 0;
ttheta.C(:) = 0;

ttheta = {ttheta};

otheta(:) = ttheta(:);
oy = tapas_mpdcm_fmri_predict(y, u, otheta, ptheta);
ollh = pars.fllh(y, u, otheta, ptheta, oy);

ollh = sum(ollh, 1);
olpp = sum(pars.flpp(y, u, otheta, ptheta), 1);

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
        if pars.verb
            fprintf(1, 'Starting llh: %0.5f\n', ollh);
        end
        op(i) = tapas_mpdcm_fmri_get_parameters(otheta, ptheta);
    catch err
        if strcmp(err.identifier, 'mpdcm:fmri:mle:numeric') || ...
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
