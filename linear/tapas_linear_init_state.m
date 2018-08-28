function [state] = tapas_linear_init_state(data, model, inference)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nc = numel(model.graph{1}.htheta.T);
np = size(data.u, 1);
nb = size(data.u{1}, 2);

state = struct('graph', [], 'llh', [], 'kernel', []);

state.graph = cell(4, 1);
state.llh = cell(4, 1);
state.kernel = cell(4, 1);

state.graph{1} = data;
state.graph{2} = struct('y', [], 'u', []);

state.graph{2}.y = cell(np, nc);
state.graph{2}.y(:) = {ones(nb, 1)};

state.graph{3} = struct('y', [], 'u', []);
state.graph{3}.y = cell(1, nc);
state.graph{3}.y(:) = {zeros(nb, 1)};

% Use the hyperpriors as state
state.graph{4} = struct('y', [], 'u', []);
state.graph{4}.y = cell(1, nc);
state.graph{4}.y(:) = {model.graph{4}.htheta};

state.llh{1} = -inf * ones(np, nc);
state.llh{2} = -inf * ones(np, nc);
state.llh{3} = -inf * ones(1, nc);

state.kernel{2} = cell(np, nc);
state.kernel{2}(:) = inference.kernel(2);

state.kernel{3} = cell(1, nc);
state.kernel{3}(:) = inference.kernel(3);
state.v = zeros(np, nc);

state.nsample = 0;

end

