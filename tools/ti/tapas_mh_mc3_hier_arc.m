function [v] = tapas_mh_mc3_hier_arc(ollh, olpp, nllh, nlpp, ...
    ratio, T)
%% Acceptance rejection criterion for metropolis hastings in the context of
% population mcmc.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

v =  sum(nllh, 1) .* T + nlpp - (sum(ollh, 1) .* T + olpp) + ratio;

nansv = isnan(v);
v(nansv) = -inf;

v = rand(size(v)) < exp(min(v, 0));

assert(all(-inf < nllh(v) + nlpp(v)), 'tapas:mh', ...
    '-inf value in the new samples');

end

