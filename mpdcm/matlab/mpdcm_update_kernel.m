function [ns, nc] = mpdcm_update_kernel(s, r, os, oc, t)
%% Updates the candidate transition kernel for MCMC.
%
% s -- Samples drawn
% r -- Rejections 
% os -- Old sigma
% oc -- Old covariance
% t -- Number of iteration
%
% ns -- New sigma
% nc -- New covariance
%
% The kernel is changed according to the algorithm proposed by 
% using the method proposed  by 
% Shaby B, Wells M (2010). "Exploring an Adaptive Metropolis Algorithm."
% 
% aponteeduardo@gmail.com
% copyright (C) 2014
%

% Optimal ratio
or = 0.234;

gamma1 = t^(-c1);
gamma2 = c0 * gamma1;

ns = size(s, 2);
pr = mean(r, 2);
nc = zeros(size(s, 1), size(s, 1), size(s, 3);

for i = 1:size(r, 1)
    nc = oc(:, :, i) + gamma1 * s(:, :, i) * s(:, :, i)';
end

ns = exp(log(os) + 0.5*gamma2*(pr - or));

end
