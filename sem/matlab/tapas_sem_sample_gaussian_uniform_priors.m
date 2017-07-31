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

% Beta parameters.
bdist = zeros(size(ptheta.pm));
bdist(ptheta.bdist) = 1;
bdist = logical(bdist .* sample_pars);
vals = betarnd(abs(ptheta.mu), abs(ptheta.pm));
vals = ptheta.jm * ptheta.sm' * vals;
theta(bdist) = log(vals(bdist) ./ ( 1 - vals(bdist)));

theta(~sample_pars) = ptheta.p0(~sample_pars);


end

