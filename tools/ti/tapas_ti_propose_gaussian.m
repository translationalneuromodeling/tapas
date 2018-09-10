function [ntheta, ratio] = tapas_ti_propose_gaussian(otheta, ptheta, htheta)
%% Makes a new proposal using a purely Gaussian kernel. 
%
%   Input
%       otheta          -- Array with old parameters.
%       ptheta          -- Priors.
%       htheta          -- Parameters of the proposal distribution. The field
%                          kn is a structure representing the kernels. 
%                          Two fields: s which is a scaling factor and S which
%                          is the Cholosvky decomposition of the kernel.

%   Output
%       ntheta          -- New proposal.
%       ratio           -- Ratio of the new proposal.
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%

kn = htheta.ok;

nt = numel(otheta);
ratio = zeros(1, nt); % Assumes the kernel has not been trucated.
ntheta = cell(size(otheta));

% Sample and project to a possibly higher dimensional space
for i = 1:nt
    rprop = randn(size(kn(i).S, 1), 1);
    rprop = htheta.mixed .* (kn(i).k' * rprop) + ...
        htheta.nmixed .* (htheta.knmixed * rprop);
    ntheta{i} = full(otheta{i} + rprop);
end


end

