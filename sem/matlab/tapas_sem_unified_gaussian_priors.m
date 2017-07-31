function [mmu, mvu, vmu, vvu, me, ve, ml, vl, p0m, p0v] = ...
    tapas_sem_unified_gaussian_priors()
%% A single set of priors in terms of mean and variance.
%
%   Input
%
%   Ouput
%       mmu     -- Mean of the mean of the units
%       mvu     -- Variance of the mean of the units
%       vmu     -- Mean of the variance of the units
%       vvu     -- Variance of the variance of the units
%       me      -- Mean of the non decision time of the early unit
%       ve      -- Variance of the non decision time of the early unit
%       ml      -- Mean of the non decision time of the late unit
%       vl      -- Variance of the non decision time of the late unit
%       p0m     -- Parameter alpha of the p0 probability
%       p0v     -- Parameter of the p0 probability

% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Mean of the units
[mmu, mvu] = norm2lognorm(0.55, 0.5); % Invert
%[mmu, mvu] = norm2lognorm(0.55, 0.05); % Simulate


% Variance of the units
%[vmu, vvu] = norm2lognorm(0.5, 0.1); % Invert
[vmu, vvu] = norm2lognorm(0.1, 0.01); % Simulate

ve = log(1.25/(0.5 * 0.5) + 1); % Mean is 0.5 and variance is 1.25
me = log(0.5) - ve/2;

vl = log(1.25/(0.75 * 0.75) + 1); % Mean is 0.75 and variance is 1.25
ml = log(0.75) - vl/2;

p0m = 0.5;
p0v = 0.5;

end

function [m, s] = norm2lognorm(mu, sig2)
    m = log(mu^2/sqrt(sig2 + mu^2));
    s = log(sig2/mu^2 + 1);
end

