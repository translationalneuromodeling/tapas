function [v] = tapas_mh_mc3g_arc(ollh, olpp, nllh, nlpp, ...
    ratio, T)
%% Acceptance rejection criterion for metropolis hastings in the context of
% population mcmc generalized for hierarchical models.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

v = nllh .* T + nlpp - (ollh .* T + olpp) + ratio;

nansv = isnan(v);
v(nansv) = -inf;

v = rand(size(v)) < exp(min(v, 0));

assert(all(-inf < nllh(v) + nlpp(v)), 'tapas:mh', ...
    '-inf value in the new samples');

end

