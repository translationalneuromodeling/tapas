function [state] = tapas_sampler_mc3_tempering(data, model, inference, state)
%% Sampler for population mcmc (mc3) with generalized temperature.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

ollh = state.llh{1};
olpp = state.llh{2};
temperature_ordering = state.graph{2}.u.temperature_ordering;


T = model.graph{1}.htheta.T;
[nv, nc] = size(T); 

% Doesn't need to do anything
if nc == 1
    return
end

c = 0;
for l = 1 : inference.mc3it
    sc = ceil(rand() * (nc - 1));
    v0 = temperature_ordering(sc); 
    v1 = temperature_ordering(sc + 1); 
    dt = T(:, sc + 1) - T(:, sc);
    p = exp(sum((ollh(:, v0) - ollh(:, v1)) .* dt));
    if rand() < p
        temperature_ordering([v0, v1]) = temperature_ordering([v1, v0]);
        %c = c + 1;
    end
end
%fprintf(1, 'Swapping rate %0.2f\n', c/inference.mc3it);
state.graph{2}.u.temperature_ordering = temperature_ordering;

end
