function [v] = tapas_mh_mc3_arc(ollh, olpp, nllh, nlpp, ...
    ratio, T)
%% Acceptance rejection criterion for metropolis hastings in the context of
% population mcmc.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

v = bsxfun(@times, nllh, T) + nlpp - (bsxfun(@times, ollh, T) + olpp) ...
    + ratio;

nansv = isnan(v);
v(nansv) = -inf;

v = rand(size(v)) < exp(v);

assert(all(-inf < nllh(v) + nlpp(v)), 'tapas:mh', ...
    '-inf value in the new samples');

end

