function [ps, fe] = tapas_mpdcm_fmri_ps(dcm, pars)
%% Estimates the posterior probability of the parameters using MCMC combined
% with path sampling.
%
% Input:
% dcm       -- Struct. DCM to be estimated
% pars      -- Struct. Parameters for the mcmc. verb: Verbose option. Defs. 
%           False. mc3i: Number of times that the exchange operator is applied.
%           Defs. 20. diagi: Number of samples before recomputing the kernel 
%           and producing verbose output.
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

if ~isfield(pars, 'verb')
    pars.verb = 0;
end

if ~isfield(pars, 'mc3i')
    pars.mc3i = 20;
end

if ~isfield(pars, 'diagi')
    pars.diagi = 200;
end

if ~isfield(pars, 'integ')
    pars.integ = 'kr4';
end

if ~isfield(pars, 'dt')
    pars.dt = 1;
end

T = sort(pars.T);
nburnin = pars.nburnin;
niter = pars.niter;

[y, u, theta, ptheta] = tapas_mpdcm_fmri_tinput(dcm);

ptheta.integ = pars.integ;
ptheta.dt = pars.dt;

htheta = tapas_mpdcm_fmri_htheta(ptheta);

nt = numel(T);

[q, otheta, ollh, olpp] = init_estimate(y, u, theta, ptheta, T, pars);

op = tapas_mpdcm_fmri_get_parameters(otheta, ptheta);

ps_theta = zeros(numel(op{end}), niter);
ellh = zeros(nt, niter);

diagnostics = zeros(1, nt);

ok = init_kernel(op, y);
os = zeros(numel(op{1}), nt, pars.diagi);

t = 1;

for i = 1:nburnin+niter

    % Diagnostics and kernel update
    if i > 1 && mod(i-1, pars.diagi) == 0
        toc
        diagnostics = diagnostics/pars.diagi;
        if pars.verb
            fprintf(1, 'Iter %d, diagnostics:  ', i);
            fprintf(1, '%0.2f ', diagnostics);
            fprintf(1, '%0.2f ', ollh);
            fprintf(1, '\n');
            if i > nburnin
                fe = trapz(T, mean(ellh(:,1:i-nburnin), 2));
                fprintf(1, 'Fe: %0.05f\n', fe);
            end
        end
        ok = update_kernel(t, ok, os, diagnostics); 
        diagnostics(:) = 0;
        t = t + 1;
        tic
    end


    np = tapas_mpdcm_fmri_sample(op, ptheta, htheta, ok);
    ntheta = tapas_mpdcm_fmri_set_parameters(np, otheta, ptheta);

    [nllh, ny] = tapas_mpdcm_fmri_llh(y, u, ntheta, ptheta, 1);

    nllh = sum(nllh, 1);
    nlpp = sum(tapas_mpdcm_fmri_lpp(y, u, ntheta, ptheta), 1);

    nllh(isnan(nllh)) = -inf;
        
    v = nllh.*T + nlpp - (ollh.*T + olpp);
    tv = v;
    v = rand(size(v)) < exp(bsxfun(@min, v, 0));

    ollh(v) = nllh(v);
    olpp(v) = nlpp(v);
    op(:, v) = np(:, v);

    assert(all(-inf < ollh), 'mpdcm:fmri:ps', '-inf value in the likelihood');

    diagnostics(:) = diagnostics(:) + v(:);

    if i > nburnin
        ps_theta(:, i - nburnin) = op{end};
        ellh(:, i - nburnin) = ollh;
    else
        os(:, :, mod(i-1, pars.diagi) + 1) = cell2mat(op);
    end


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
fe = trapz(T, mean(ellh, 2));

ps.pE = mean(ps_theta , 2);
ps.theta = tapas_mpdcm_fmri_set_parameters({ps.pE}, theta(1), ptheta);
ps.y = tapas_mpdcm_fmri_int(u, ps.theta, ptheta);
ps.theta = ps_theta;
ps.y = ps.y{:};
ps.F = fe;
% Prior posterior mean
ps.ppm = 1/mean(exp(-ellh(end, :))); 
% Prior arithmetic mean (or almost)
ps.pam = mean(exp(ellh(1, :))); 

end

function [nk] = init_kernel(op, y)
%% Initilize the kernel or covariance matrix of the proposal distribution.
%
% See Exploring an adaptative Metropolis Algorithm
% 

np = size(op{1}, 1);
nr = size(y{1}, 1);

c = diag([ones(1, np - (3 * nr + 1)), 1e-2 * ones(1, 2 * nr +1), ones(1, nr)]);

nk = cell(size(op, 2), 1);
nk(:) = {c};

nk = struct('S', nk, 's', 0.1);

end

function [nk] = update_kernel(t, ok, os, ar)
%% Computes a new kernel or covariance for the proposal distribution.
%
% ok -- Old kernel
% os -- Old samples
% ar -- Acceptance rate
%
% See Exploring an adaptative Metropolis Algorithm
% 


c0 = 1.0;
c1 = 0.8;

gammaS = t^-c1;
gammas = c0*gammaS; 

ns = size(os, 3);
nd = size(os, 1);
nk = ok;

% Optimal log rejection rate
ropt = 0.234;

for i = 1:numel(ok)
    % From Cholesky form to covariance form
    ok(i).S = ok(i).S * ok(i).S';
    % Empirical variance
    ts = squeeze(os(:, i, :));
    ts = bsxfun(@minus, ts, mean(ts, 2));
    ek = (ts * ts')./(ns-1);
    % Set new kernel
    nk(i).S = ok(i).S + gammaS * ( ek - ok(i).S);
    % Compute the Cholesky decomposition 
    nk(i).S = chol(nk(i).S + eye(nd) * 0.0001);
    % Set new scaling
    nk(i).s = exp(log(ok(i).s) + gammas * (ar(i) - ropt));
end
    
end

function [q, otheta, ollh, olpp] = init_estimate(y, u, theta, ptheta, T, pars)
% Create an initial estimate to reduce burn in phase by computing the 
% posterior distribution of the power posteriors

nt = numel(T);

[q, otheta] = tapas_mpdcm_fmri_gmodel(y, u, theta, ptheta);
[ollh, ny] = tapas_mpdcm_fmri_llh(y, u, otheta, ptheta);
if pars.verb
    fprintf(1, 'Starting llh: %0.5d\n', ollh);
end

[op] = tapas_mpdcm_fmri_get_parameters(otheta, ptheta);

% This is purely heuristics. There is an interpolation between the prior and
% the mle estimator such that not all chains are forced into high llh regions.
% Moreover, at low temperatures the chains are started in more sensible regime

op = op{1};
np = [(linspace(0, 1, nt)').^5 (1-linspace(0, 1, nt)).^5'] * ...
    [op ptheta.p.theta.mu]';
np = mat2cell(np', numel(op), ones(1, nt));

% Samples from posterior

theta = cell(1, nt);
theta(:) = otheta;
op = np;

% Fully initilize

otheta = tapas_mpdcm_fmri_set_parameters(op, theta, ptheta);

[ollh, ~] = tapas_mpdcm_fmri_llh(y, u, otheta, ptheta);
olpp = tapas_mpdcm_fmri_lpp(y, u, otheta, ptheta);

% Eliminate weird cases

[~, l] = min(abs(bsxfun(@minus, find(isnan(ollh)), find(~isnan(ollh))')));
tl = find(~isnan(ollh));
l = tl(l);

olpp(isnan(ollh)) = olpp(l);
op(:, isnan(ollh)) = op(:, l);
ollh(isnan(ollh)) = ollh(l);

end
