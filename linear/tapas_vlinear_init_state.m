function [state] = tapas_vlinear_init_state(data, model, inference)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

[state] = tapas_linear_init_state(data, model, inference);

nc = numel(model.graph{1}.htheta.T);

% No need of kernels because of the Gibbs step
state.kernel{3} = [];

% The state now has a mean and precision terms.
state.graph{3} = struct('y', [], 'u', []);
state.graph{3}.y = cell(1, nc);
node = struct('mu', [], 'pe', 1);
node.mu = model.graph{4}.htheta.y.mu;
state.graph{3}.y(:) = {node};

state.graph{4} = struct('y', [], 'u', []);
state.graph{4}.y = cell(1, nc);
state.graph{4}.y(:) = {model.graph{4}.htheta.y};


end
