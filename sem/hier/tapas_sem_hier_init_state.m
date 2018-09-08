function [state] = tapas_sem_hier_init_state(data, model, inference)
%% Generate the structure of the states of the sampler.
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

nc = size(model.graph{1}.htheta.T, 2);
ns = size(data, 1);
mu = model.graph{4}.htheta.y.mu;
nb = numel(mu);

state = struct('graph', [], 'llh', [], 'kernel', [], 'T', []);

state.graph = cell(4, 1);
state.llh = cell(4, 1);
state.kernel = cell(4, 1);
state.T = cell(4, 1);

state.graph{1} = data;
state.graph{2} = struct('y', [], ...
    'u', struct('temperature_ordering', uint16(1:nc)));

state.graph{2}.y = cell(ns, nc);
state.graph{2}.y(:) = {mu};

state.graph{3} = struct('y', [], 'u', []);
state.graph{3}.y = cell(1, nc);
state.graph{3}.y(:) = {struct('mu', mu, 'pe', ones(nb, 1))};

% Use the hyperpriors as state
state.graph{4} = struct('y', [], 'u', []);
state.graph{4}.y = cell(1, nc);
state.graph{4}.y(:) = {model.graph{4}.htheta.y};

state.llh{1} = -inf * ones(ns, nc);
state.llh{2} = -inf * ones(ns, nc);
state.llh{3} = -inf * ones(1, nc);

state.kernel{2} = cell(ns, nc);
state.kernel{2}(:) = inference.kernel(2);

state.T{1} = model.graph{1}.htheta.T;

state.v = zeros(ns, nc);

state.nsample = 0;

end
