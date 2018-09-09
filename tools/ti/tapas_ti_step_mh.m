function [ox, otheta, ollh, olpp, v] = tapas_ti_step_mh(y, u, ox, otheta, ...
    ollh, olpp, ptheta, htheta, pars)
%% Estimates the posterior probability of the parameters using MCMC combined
% with path sampling.
%
% Input
%   y       Saccade data.
%   u       Experimental input.
%   ptheta  Priors of the model
%   htheta  Kernel transformation.
%   pars    Parameters for the MCMC. 
%           pars.verbose   True or false. Defs. False.
%           pars.mc3it      Number of iterations for MC3. Defs. 0.
%           pars.kup        Number of steps for the kernel update. 
%                           Defs. 500.
%           pars.seed       Seed for the random generation. If zero use
%                           rng('shuffle'). Defaults to zero.
%           pars.samples    If true, stores the samples from the chain at 
%                           lowest temperature. Defaults to zero.
%
% Output
%   ps      Samples from the posterior distribuion
%   fe      Free energy estimate.
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

if isfield(ptheta, 'mh_propose_sample') 
    propose_sample = htheta.mh_propose_sample;
else
    propose_sample = @tapas_ti_propose_gaussian;
end

[ntheta, ratio] = propose_sample(otheta, ptheta, htheta);
[nllh, nx] = ptheta.llh(y, [], u, ntheta, ptheta);

T = ptheta.T;


nllh = sum(nllh, 1);
nlpp = sum(ptheta.lpp(y, nx, u, ntheta, ptheta), 1);

v = nllh .* T + nlpp - (ollh .* T + olpp) + ratio;
nansv = isnan(v);
v(nansv) = -inf;

v = rand(size(v)) < exp(bsxfun(@min, v, 0));

ollh(v) = nllh(v);
olpp(v) = nlpp(v);
otheta(:, v) = ntheta(:, v);
ox(:, v) = nx(:, v);

assert(all(-inf < ollh), 'mpdcm:ti:mh', '-inf value in the new samples');

end

