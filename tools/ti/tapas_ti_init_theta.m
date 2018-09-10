function [ntheta] = tapas_ti_init_theta(ptheta, htheta, pars)
%% Initial values of theta with the prior expected value
%
% Input 
%   T - Temperatures
%   ptheta - Priors
% 
% Output
%   ntheta -- Initialized structure array.

if isfield(ptheta, 'p0')
    mu = ptheta.p0;
else
    mu = ptheta.mu;
end

ntheta = cell(1, numel(pars.T));
ntheta(:) = {mu};


end
