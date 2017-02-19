function [llh] = tapas_sem_prosa_hier_llh(y, u, theta, ptheta)
%% Computes the likelihood function of a hierarchical model.
%
% Input 
%
%   y -- Observed behavioral data. A structure with fields 't' times and 'a' 
%       action
%   u -- Experimental input. A structure with fields: 'tt' trial type, either
%       prosaccade or antisaccade
%   theta -- Model parameters
%   ptheta -- Priors
%
% Output
%
%   llh -- Log likelilhood
% aponteeduardo@gmail.com
% copyright (C) 2015
%

DIM_THETA = tapas_sem_prosa_ndims();

llh = zeros(1, numel(theta));
vt = ~y.i;

% Blocks
b = ptheta.blocks;
nb = numel(b);
method = ptheta.method;

for i = 1:numel(theta)
    for j = 1:nb
        ltheta = theta{i}(DIM_THETA * (j - 1) + 1: j * DIM_THETA);
        ltheta = exp(ltheta);
        t = u.b == b(j) & vt;
        llh(i) = llh(i) + sum(tapas_sem_prosa_cllh(y.t(t), y.a(t), u.tt(t), ...
            ltheta, method));
    end
end

end
