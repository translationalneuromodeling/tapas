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
    pars.verbose = 1;
end

DIAGN = 50;

T = pars.T;
nburnin = pars.nburnin;
niter = pars.niter;

nt = numel(T);

[y0, u0, theta0, ptheta] = mpdcm_fmri_tinput(dcm);
htheta = mpdcm_fmri_htheta(ptheta);

y = {y0};
u = {u0};
otheta = {theta0};

[q, otheta, ollh] = init_estimate(y, u, otheta, ptheta, T);
[op] = mpdcm_fmri_get_parameters(otheta, ptheta);

olpp = mpdcm_fmri_lpp(y, u, otheta, ptheta);

% Posterior

ps_theta = zeros(numel(op{end}), niter);
ellh = zeros(nt, niter);

diagnostics = zeros(1, nt);
% Optimized kernel
kt = ones(1, nt);
tic
for i = 1:nburnin+niter

    if mod(i, DIAGN) == 0
        diagnostics = diagnostics/DIAGN;
        if pars.verbose
            fprintf(1, 'Iter %d, diagnostics:  ', i);
            fprintf(1, '%0.2f ', diagnostics);
            fprintf(1, '%0.2f ', ollh);
            fprintf(1, '\n');
        end
        if i < nburnin
            kt(diagnostics < 0.4) = kt(diagnostics < 0.4)/2;
            kt(diagnostics > 0.55) = kt(diagnostics > 0.55)*1.8;
        else
            fe = trapz(T, mean(ellh(:,1:i-nburnin), 2));
            fprintf(1, 'Estimated fe: %0.5d', fe);
        end
        diagnostics(:) = 0;
    toc
    tic
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
        % Choose a random pair
        k = ceil((numel(T)-1)*rand);
        tt = (ollh(k+1)*T(k) + ollh(k)*T(k+1)) -...
            (ollh(k)*T(k) + ollh(k+1)*T(k+1));
        if rand > min(1, exp(tt))
            op(:,[k, k+1]) = op(:, [k+1, k]);
            ollh(:,[k, k+1]) = ollh(:, [k+1, k]);
            olpp(:,[k, k+1]) = olpp(:, [k+1, k]);
        end
    end

end

fe = trapz(T, mean(ellh, 2));

ps.pE = mean(op{end} , 2);
ps.theta = mpdcm_fmri_set_parameters(op(:), {theta0}, ptheta);
ps.y = mpdcm_fmri_int(u, ps.theta, ptheta);
ps.theta = ps.theta{:};
ps.y = ps.y{:};
ps.F = fe;
end

function [q, theta, ollh] = init_estimate(y0, u0, theta0, ptheta, T)
% Create an initial estimate to reduce burn in phase by computing the 
% posterior distribution of the power posteriors

nt = numel(T);

y = cell(nt, 1);
y(:) = y0;
u = cell(nt, 1);
u(:) = u0;
theta = cell(nt, 1);
theta(:) = theta0;

ptheta.T = T;

[q, theta] = mpdcm_fmri_gmodel(y, u, theta, ptheta);
q = q';
theta = theta';

[ollh, ~] = mpdcm_fmri_llh(y0, u0, theta, ptheta);
fprintf(1, 'Starting llh: ');
fprintf(1, '%0.5d, ', ollh);
fpritnf(1, '\n');

end
