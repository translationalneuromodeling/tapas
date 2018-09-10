function [nstate] = tapas_mh_mc3_tempering_sample_node(data, model, ...
    inference, state, node)
%% Samples for a particular node assuming there is a function for it. 
% Fundamentally the temperature schedule is assumed to a state and not a
% parameter.
%
%
% This works as the conditional independency of the level allows one to sample
% without worrying about the values in other nodes.

%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nstate = state;

% Theta is a data object
theta = inference.mh_sampler{node}.propose_sample(data, model, inference, ...
    state, node);

y = state.graph{node - 1};

nllh = model.graph{node - 1}.llh(y, theta, model.graph{node -  1}.htheta);
nlpp = model.graph{node}.llh(theta, state.graph{node + 1}, ...
    model.graph{node}.htheta);

% Reorder the temperature before passing it to the sampler
nt = numel(state.graph{node}.u.temperature_ordering);
nv = zeros(size(state.graph{node}.u.temperature_ordering));
for i = 1:nt
    nv(state.graph{node}.u.temperature_ordering(i)) = i;
end
%T = state.T{node - 1}(:, state.graph{node}.u.temperature_ordering);
T = state.T{node - 1}(:, nv);

[v] = inference.mh_sampler{node}.ar_rule(state.llh{node - 1}, ...
    state.llh{node}, nllh, nlpp, 0, T);

nstate.graph{node}.y(v) = theta.y(v);
nstate.llh{node - 1}(v) = nllh(v);
nstate.llh{node}(v) = nlpp(v);

nstate.v = v;
nstate.nsample = state.nsample + 1;

end
