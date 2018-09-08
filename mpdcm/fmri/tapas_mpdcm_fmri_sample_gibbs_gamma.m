function [otheta, ny, ollh, olpp] = tapas_mpdcm_fmri_sample_gibbs_gamma(...
    y, u, otheta, ptheta, pars, ny, ollh, olpp)
%% Makes a Gibbs step of a set of the parameters.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

T = pars.T;
nt = numel(T);

% Perfom a gibb step on the residuals
[ntheta, oq, nq, olp, nlp, nllh] = draw_samples(y, u, otheta, ptheta, ny, ollh, pars);

% Make an extra MH step to adjust
% Because of the conjugacy one could obviate the likelihood term but anyways
% one has to compute it without having to recompute the signal.
v = (nlp + oq) - (olp + nq);
v = v > log(rand(1, nt));

%fprintf(1, '%d', v);
%fprintf(1, '   %0.3f', mean(v));
%fprintf(1, '\n');

otheta(v) = ntheta(v);

% Save computing the likelihood
ollh(v) = nllh(v);
% Spare computing the prior 
olpp(v) = olpp(v) - olp(v) + nlp(v);

assert(all(-inf < ollh), 'mpdcm:fmri:ps', ...
    'gibbs -inf value in the likelihood');

end % 

function [theta, oq, nq, olpp, nlpp, nllh] = draw_samples(y, u, ...
    theta, ptheta, ny, ollh, pars)
%% Draw a sample for lthe distribution

nt = numel(theta);
nr = size(y{1}, 1);
nd = size(y{1}, 2);
nb = size(ptheta.X0, 2);

y = y{1}';

oq = zeros(1, nt);
nq = zeros(1, nt);
nlpp = zeros(1, nt);
olpp = zeros(1, nt);

% Adjust the priors to mimic the log normal prior
a0 = 1.14;
b0 = 0.61;

% Get the index of the lambda parameters
i0 = numel(ptheta.p.theta.mu) - nb * nr - nr + 1;
i1 = numel(ptheta.p.theta.mu) - nb * nr;

mu = ptheta.p.theta.mu(i0:i1);
pe = diag(ptheta.p.theta.pi);
pe = pe(i0:i1);

%b0 = 1./(exp(mu + 1.5 ./ pe) - exp(mu + 0.5./ pe));
%a0 = exp(mu + 0.5 ./ pe) .* b0;

nllh = zeros(size(ollh));

for i = 1:nt
    % Assume no correlations between regions i.e., treat the problem as 
    % massive multivariate problem

    %if any(isnan(ny{i}(:))) || any(isinf(ny{i}(:)))
    %    continue
    %end
    % Fit only the residual
    if numel(ptheta.X0)
        r = y - (ny{i} + ptheta.X0 * theta{i}.beta);
    else 
        r = y - ny{i};
    end
    
    r2 = sum(r.*r, 1)';
    ta = a0 + 0.5 * nd * ones(nr, 1) * pars.T(i);
    tb = b0 + 0.5 * r2 * pars.T(i);

    olambda = theta{i}.lambda;
    oq(i) = sum((a0 - 1) .* olambda - exp(olambda) .* b0);
    nlambda = log(randg(ta)) - log(tb);
    nq(i) = sum((a0 - 1) .* nlambda - exp(nlambda) .* b0);
     
    theta{i}.lambda = nlambda;

    olpp(i) = -sum(0.5 * (olambda - mu) .* (olambda - mu) .* pe ...
        + olambda);
    nlpp(i) = -sum(0.5 * (nlambda - mu) .* (nlambda - mu) .* pe ...
        + nlambda);

    % Save computing the likelihood
    nllh(i) = ollh(i) - ...
        0.5 * sum(-r2 .* exp(olambda) + nd * olambda) + ...
        0.5 * sum(-r2 .* exp(nlambda) + nd * nlambda);
    
end

end
