function [llh] = tapas_sem_seri_llh(y, u, theta, ptheta)
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

llh = zeros(size(theta));
it = y.i;

method = ptheta.method;
ptrans = ptheta.ptrans;

for i = 1:numel(theta)
    ttheta = ptrans(theta{i});
    llh(i) = sum(tapas_sem_seri_cllh(y.t(~it), y.a(~it), u.tt(~it), ttheta, ...
        method, 1), 1);
end


end

