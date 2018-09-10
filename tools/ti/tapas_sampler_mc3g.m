function [state] = tapas_sampler_mc3g(data, model, inference, state)
%% Sampler for population mcmc (mc3) with generalized temperature.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

ollh = state.llh{1};
olpp = state.llh{2};

% Get the joint 
%for i = 2:nn - 1
%    lpp = lpp + sum(state.llh{i}, 1);
%end

T = model.graph{1}.htheta.T;
[nv, nc] = size(T); 
v = reshape([1:nv * nc], nv, nc);

% Doesn't need to do anything
if nc == 1
    return
end

c = 0;
for l = 1 : inference.mc3it
    s1 = ceil(rand() * nv);
    s2 = ceil(rand() * (nc - 1));
    % Realinate
    [s1, s2i] = ind2sub([nv, nc], v(s1, s2));
    [s1, s2j] = ind2sub([nv, nc], v(s1, s2 + 1));
    
    % Now we need to adjust the probability
    plpp1 = model.graph{2}.llh_sn(state.graph{2}, ...
        state.graph{3}, model.graph{2}.htheta, s1, s2, s2j);
    plpp2 = model.graph{2}.llh_sn(state.graph{2}, ...
        state.graph{3}, model.graph{2}.htheta, s1, s2 + 1, s2i);

    % Compute the ratio
    p = exp((ollh(s1, s2i) .* T(s1, s2 + 1) + ...
        ollh(s1, s2j) .* T(s1, s2) + ...
        plpp1 + plpp2) - ...
        (ollh(s1, s2i) .* T(s1, s2) + ...
        ollh(s1, s2j) .* T(s1, s2 + 1) + ...
        olpp(s1, s2) + olpp(s1, s2 + 1) ));
    if rand() < p
        c = c + 1;
        v(s1, [s2, s2 + 1]) = v(s1, [s2 + 1, s2]);
        olpp(s1, s2) = plpp1;
        olpp(s1, s2 + 1) = plpp2;
    end
end
%:fprintf(1, 'Swap rate %0.5f\n', c/inference.mc3it);
state.llh{1} = state.llh{1}(v);
state.llh{2} = olpp;
state.graph{2}.y = state.graph{2}.y(v);

end
