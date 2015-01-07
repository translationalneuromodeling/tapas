function [ps, fe] = mpdcm_fmri_ps(dcm, pars)
%% Estimates the posterior probability of the parameters using MCMC combined
% with path sampling.
% dcm -- DCM to be estimated
% pars -- Parameters for the mcmc
%
% Uses a standard MCMC on a population and applies an exchange operator
% to improve the mixing.
%
%
% Real parameter evolutionary Monte Carlo with applications to Bayes Mixture
% Models, Journal of American Statistica Association, Liang & Wong 2001.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if ~isfield(pars, 'verbose')
    pars.verbose = 0;
end

DIAGN = 200;

T = pars.T;
nburnin = pars.nburnin;
niter = pars.niter;

[y, u, theta, ptheta] = mpdcm_fmri_tinput(dcm);

htheta = mpdcm_fmri_htheta(ptheta);

nt = numel(T);

[q, otheta, ollh, olpp] = init_estimate(y, u, theta, ptheta, T);

op = mpdcm_fmri_get_parameters(otheta, ptheta);

ps_theta = zeros(numel(op{end}), niter);
ellh = zeros(nt, niter);

diagnostics = zeros(1, nt);

% Optimized kernel
kt = ones(1, nt);

for i = 1:nburnin+niter

    if mod(i, DIAGN) == 0
        diagnostics = diagnostics/DIAGN;
        if pars.verbose
            fprintf(1, 'Iter %d, diagnostics:  ', i);
            fprintf(1, '%0.2f ', diagnostics);
            fprintf(1, '%0.2f ', ollh);
            fprintf(1, '\n');
            if i > nburnin
                fe = trapz(T, mean(ellh(:,1:i-nburnin), 2));
                fprintf(1, 'Fe: %0.05f\n', fe);
            end
        end
        if i < nburnin
            kt(diagnostics < 0.2) = kt(diagnostics < 0.2)/2;
            kt(diagnostics > 0.3) = kt(diagnostics > 0.3)*1.8;
        end
        diagnostics(:) = 0;
    end

    np = mpdcm_fmri_sample(op, ptheta, htheta, num2cell(kt));
    ntheta = mpdcm_fmri_set_parameters(np, otheta, ptheta);

    [nllh, ny] = mpdcm_fmri_llh(y, u, ntheta, ptheta, 1);

    nllh = sum(nllh, 1);
    nlpp = sum(mpdcm_fmri_lpp(y, u, ntheta, ptheta), 1);

    nllh(isnan(nllh)) = -inf;

    v = nllh.*T + nlpp - (ollh.*T + olpp);
    tv = v;
    v = rand(size(v)) < exp(bsxfun(@min, v, 0));

    ollh(v) = nllh(v);
    olpp(v) = nlpp(v);
    op(:, v) = np(:, v);

    diagnostics(:) = diagnostics(:) + v(:);

    if i > nburnin
        ps_theta(:, i - nburnin) = op{end};
        ellh(:, i - nburnin) = ollh;
    end

    assert(all(-inf < ollh), 'mpdcm:fmri:ps', '-inf value in the likelihood');

    if pars.mc3
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

fe = trapz(T, mean(ellh, 2));

ps.pE = mean(op{end} , 2);
ps.theta = mpdcm_fmri_set_parameters(op(:), theta, ptheta);
ps.y = mpdcm_fmri_int(u, ps.theta, ptheta);
ps.theta = ps.theta{:};
ps.y = ps.y{:};
ps.F = fe;

end


function [q, otheta, ollh, olpp] = init_estimate(y, u, theta, ptheta, T)
% Create an initial estimate to reduce burn in phase by computing the 
% posterior distribution of the power posteriors

nt = numel(T);

[q, otheta] = mpdcm_fmri_gmodel(y, u, theta, ptheta);
[ollh, ny] = mpdcm_fmri_llh(y, u, otheta, ptheta);
fprintf(1, 'Starting llh: %0.5d\n', ollh);

[op] = mpdcm_fmri_get_parameters(otheta, ptheta);

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

otheta = mpdcm_fmri_set_parameters(op, theta, ptheta);

[ollh, ~] = mpdcm_fmri_llh(y, u, otheta, ptheta);
olpp = mpdcm_fmri_lpp(y, u, otheta, ptheta);

% Eliminate weird cases

[~, l] = min(abs(bsxfun(@minus, find(isnan(ollh)), find(~isnan(ollh))')));
tl = find(~isnan(ollh));
l = tl(l);

olpp(isnan(ollh)) = olpp(l);
op(:, isnan(ollh)) = op(:, l);
ollh(isnan(ollh)) = ollh(l);

end
