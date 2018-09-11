function [model] = tapas_h2gf_model(data, hgf, pars)
%% Get the model from the hgf.
%
% Input
%       hgf         -- Hgf model.
%       pars        -- pars structure.
% Output
%       model       -- Model structre.
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%
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

model.graph{1}.llh = @tapas_h2gf_llh;
model.graph{2}.llh = @tapas_dlinear_hier_llh;
model.graph{3}.llh = @tapas_dlinear_llh;
model.graph{4}.llh = [];

% Needed for efficient parallel tempering
model.graph{2}.llh_sn = @tapas_dlinear_hier_llh_sn;

model.graph{1}.htheta = struct('pe', 0.5, 'T', pars.T, 'hgf', hgf);

model.graph{2}.htheta = struct('T', ones(size(pars.T, 2)));
model.graph{3}.htheta = struct('T', ones(size(pars.T, 2)));

% The last level is a dummy used to store the hyperpriors.

model.graph{4}.htheta = struct('y', [], 'u', []);

end

