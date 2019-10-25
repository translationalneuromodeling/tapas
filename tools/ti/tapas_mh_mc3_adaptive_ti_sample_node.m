function [nstate] = tapas_mh_mc3_adaptive_ti_sample_node(data, model, ...
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

% Theta is a data object (proposal of new parameters for all subjects)
theta = inference.mh_sampler{node}.propose_sample(data, model, inference, ...
    state, node);

y = state.graph{node - 1};

% New log-likelihood
nllh = model.graph{node - 1}.llh(y, theta, model.graph{node -  1}.htheta);
% New log-probability under the prior
nlpp = model.graph{node}.llh(theta, state.graph{node + 1}, ...
    model.graph{node}.htheta);

% Accept or reject proposals
% COMMENT: go through this.
[v] = inference.mh_sampler{node}.ar_rule(state.llh{node - 1}, ...
    state.llh{node}, nllh, nlpp, 0, state.T{node - 1});

nstate.graph{node}.y(v) = theta.y(v);
nstate.llh{node - 1}(v) = nllh(v);
nstate.llh{node}(v) = nlpp(v);

nstate.v = v;
nstate.nsample = state.nsample + 1;

end

