function [ps_theta, fe] = mpdcm_fmri_ps(dcm, pars)
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

T = pars.T;
nburnin = pars.nburnin;
niter = pars.niter;

nt = numel(T);

[y0, u0, theta0, ptheta] = mpdcm_fmri_tinput(dcm);
htheta = mpdcm_fmri_htheta(ptheta);

u = {u0};
y = {y0};

% Samples from posterior

theta = cell(1, numel(T));
theta(:) = {theta0};

op = mpdcm_fmri_get_parameters(theta, ptheta);
op = mpdcm_fmri_sample(op, ptheta, htheta);

otheta = mpdcm_fmri_set_parameters(op, theta, ptheta);

ollh = mpdcm_fmri_llh(y, u, otheta, ptheta);
olpp = mpdcm_fmri_lpp(y, u, otheta, ptheta);

ollh(isnan(ollh)) = -inf;

ps_theta = zeros(numel(spm_vec(op)), niter);
ellh = zeros(numel(T), niter);

start = 0;

for i = 1:nburnin+niter

    if mod(i, 10) == 0
        fprintf(1, 'Iteration: %d; llh: %0.5f\n', i, ollh(end));
    end

    np = mpdcm_fmri_sample(op, ptheta, htheta);
    ntheta = mpdcm_fmri_set_parameters(np, theta, ptheta);
    
    nllh = sum(mpdcm_fmri_llh(y, u, ntheta, ptheta, 1), 1);
    nlpp = sum(mpdcm_fmri_lpp(y, u, ntheta, ptheta), 1);

    nllh(isnan(nllh)) = -inf;

    v = nllh.*T + nlpp - ollh.*T - olpp;
    v = rand(size(v)) < exp(bsxfun(@min, v, 0));

    ollh(v) = nllh(v);
    olpp(v) = nlpp(v);
    op(:, v) = np(:, v);

    % Start the likelihood function in a stable point
    if ~start
        v = find(nllh > -inf);
        if any(v(:))
            op(:) = np(:, v(1));
            ollh(:) = ollh(v(1));
            olpp(:) = olpp(v(1));
            start = 1;
        end
    end

    if i > nburnin
        ps_theta(:, i - nburnin) = spm_vec(op);
        ellh(:, i - nburnin) = ollh;
    end

    % Apply an exchange operator

    k = ceil(rand() * (nt - 1) );
    ev =  [ollh(k+1) - ollh(k) ollh(k) - ollh(k+1)] *T(k:k+1)';
    if exp(ev) > rand()
        ollh(k:k+1) = ollh(k+1:-1:k);
        op(:, k:k+1) = op(:, k+1:-1:k);
    end

end

fe = trapz(diff([0 T]), mean(ellh, 2));


end
