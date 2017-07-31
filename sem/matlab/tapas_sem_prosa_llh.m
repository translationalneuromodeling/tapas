function [llh] = tapas_sem_prosa_llh(y, u, theta, ptheta)
%% Computes the likelihood of the data.
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
%
% aponteeduardo@gmail.com
% copyright (C) 2015
%

% Compute the likelihood of antisaccades and prosaccades

llh = zeros(1, numel(theta));

it = y.i;

method = ptheta.method;
ptrans = ptheta.ptrans;

for i = 1:numel(theta)
    llh(i) = sum(tapas_sem_prosa_cllh(y.t(~it), y.a(~it), u.tt(~it), ...
        ptrans(theta{i}), method, 1), 1);
end


end

