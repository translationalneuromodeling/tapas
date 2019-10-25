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

% Likelihood (individual level)
model.graph{1}.llh = @tapas_h2gf_llh;
% Parameters (individual level)
model.graph{2}.llh = @tapas_dlinear_hier_llh;
% Parameters (population level)
model.graph{3}.llh = @tapas_dlinear_llh;
% Placeholder for hyperparameters (fixed)
model.graph{4}.llh = [];

% Needed for efficient parallel tempering
model.graph{2}.llh_sn = @tapas_dlinear_hier_llh_sn;

% For 
model.graph{1}.htheta = struct('T', pars.T, 'hgf', hgf);
model.graph{2}.htheta = struct('T', ones(size(pars.T, 2)));
model.graph{3}.htheta = struct('T', ones(size(pars.T, 2)));

% The last level is a dummy used to store the hyperpriors.
% y: variable the likelihood is defined on, llh = p(y|y_above, u)
% u: parameters of likelihood for y (fixed)
model.graph{4}.htheta = struct('y', [], 'u', []);

end

