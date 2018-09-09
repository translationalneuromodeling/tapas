function [model] = tapas_linear_model(model, pars)
%% Define the model.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

model = struct('graph', []);
model.graph = cell(4, 1);

for i = 1:4
    model.graph{i} = struct('llh', [], 'htheta', []);
end

model.graph{1}.llh = @tapas_linear_llh;
model.graph{2}.llh = @tapas_linear_hier_llh;
model.graph{3}.llh = @tapas_linear_hier_llh;
model.graph{4}.llh = [];

model.graph{1}.htheta = struct('pe', 0.5, 'T', pars.T);
model.graph{2}.htheta = struct('pe', 0.5, 'T', ones(size(pars.T)));
model.graph{3}.htheta = struct('pe', 0.5, 'T', ones(size(pars.T)));

% The last level is a dummy used to store the hyperpriors.

model.graph{4}.htheta = struct('y', [], 'u', []);


end

