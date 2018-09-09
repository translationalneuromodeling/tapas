function [state] = tapas_sem_multiv_init_state(data, model, inference)
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
np = size(model.graph{1}.htheta.model.jm, 2);

state = struct('graph', [], 'llh', [], 'kernel', [], 'T', [], 'time', tic);

state.graph = cell(4, 1);
state.llh = cell(4, 1);
state.kernel = cell(4, 1);
state.T = cell(4, 1);

state.graph{1} = data;
state.graph{2} = struct('y', [], 'u', []);

state.graph{2}.y = cell(ns, nc);
ty = model.graph{1}.htheta.model.x * model.graph{4}.htheta.y.mu;
ty = reshape(ty', numel(ty), 1);
state.graph{2}.y = repmat(mat2cell(ty, np * ones(ns, 1), 1), 1, nc);
% Regressors
state.graph{2}.u = struct(...
    'x', model.graph{1}.htheta.model.x, ...
    'omega', model.graph{1}.htheta.model.omega, ... x'x + I
    'iomega', model.graph{1}.htheta.model.iomega, ... inv(x'x + I)
    'comega', model.graph{1}.htheta.model.comega, ...
    'ciomega', model.graph{1}.htheta.model.ciomega, ...
    'temperature_ordering', uint16(1:nc));

state.graph{3} = struct('y', [], 'u', []);
state.graph{3}.y = cell(1, nc);
state.graph{3}.y(:) = {struct('mu', mu, 'pe', ones(1, np))};

% Use the hyperpriors as state
state.graph{4} = struct('y', struct('mu', []), 'u', []);
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
