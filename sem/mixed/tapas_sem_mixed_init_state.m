function [state] = tapas_sem_mixed_init_state(data, model, inference)
%% Generate the structure of the states of the sampler.
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

ptheta = model.graph{1}.htheta.model;

nc = size(model.graph{1}.htheta.T, 2);
ns = size(data, 1);
mu = model.graph{4}.htheta.y.mu;
nb = numel(mu);
np = size(ptheta.jm, 2);

state = struct('graph', [], 'llh', [], 'kernel', [], 'T', []);

state.graph = cell(5, 1);
state.llh = cell(5, 1);
state.kernel = cell(5, 1);
state.T = cell(5, 1);

% First node
state.graph{1} = data;

% Second node
state.graph{2} = struct('y', [], 'u', []);
state.graph{2}.y = cell(ns, nc);
ty = ptheta.x * model.graph{4}.htheta.y.mu;
ty = reshape(ty', numel(ty), 1);
state.graph{2}.y = repmat(mat2cell(ty, np * ones(ns, 1), 1), 1, nc);
% Regressors
state.graph{2}.u = struct(...
    'x', ptheta.x, ...
    'omega', ptheta.omega, ... x'x + I
    'iomega', ptheta.iomega, ... inv(x'x + I)
    'comega', ptheta.comega, ...
    'ciomega', ptheta.ciomega); % sqrt(omega)

% Third node
state.graph{3} = struct('y', [], 'u', []);
state.graph{3}.y = cell(1, nc);
state.graph{3}.y(:) = {model.graph{3}.htheta.y};
state.graph{3}.u = model.graph{3}.htheta.u;

% Fourth node, 
state.graph{4} = struct('y', [], 'u', []);
state.graph{4}.y = cell(1, nc);
state.graph{4}.y(:) = {model.graph{4}.htheta.y};
state.graph{4}.u = model.graph{4}.htheta.u;

% Fith node
state.graph{5} = struct('y', [], 'u', []);
state.graph{5}.y = cell(1, nc);
state.graph{5}.y(:) = {model.graph{5}.htheta.y};
state.graph{5}.u = model.graph{5}.htheta.u;

% Likelihood
state.llh{1} = -inf * ones(ns, nc);
state.llh{2} = -inf * ones(ns, nc);
state.llh{3} = -inf * ones(1, nc);
state.llh{4} = -inf * ones(1, nc);

state.kernel{2} = cell(ns, nc);
state.kernel{2}(:) = inference.kernel(2);

state.T{1} = model.graph{1}.htheta.T;

state.v = zeros(ns, nc);

state.nsample = 0;

end
