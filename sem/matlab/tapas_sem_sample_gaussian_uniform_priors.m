function [theta] = tapas_sem_sample_gaussian_uniform_priors(ptheta)
%% Sample from a Gaussian prior. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

np = size(ptheta.jm, 2);

sample_pars = logical(sum(ptheta.jm, 2));

if size(ptheta.pm, 2) ~= np
    pm = diag(ptheta.pm);
else
    pm = ptheta.pm;
end

lt = chol(pm);
theta = ptheta.mu +  lt \ ptheta.jm * randn(np, 1);
theta(~sample_pars) = ptheta.p0(~sample_pars);

% Uniform parameters

rates = ptheta.jm * tan(pi * (rand(np, 1) - 0.5));
rates(~sample_pars) = ptheta.p0(~sample_pars);

index = ptheta.bdist;
theta(index) = rates(index);

end

