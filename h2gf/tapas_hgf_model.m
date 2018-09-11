function [model] = tapas_hgf_model(hgf, pars)
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

model.graph{1}.llh = @tapas_hgf_llh;
model.graph{2}.llh = @tapas_vlinear_llh;
model.graph{3}.llh = [];

model.graph{1}.htheta = struct('pe', 0.5, 'T', pars.T);
model.graph{2}.htheta = struct('T', ones(size(pars.T)));

% The last level is a dummy used to store the hyperpriors.

model.graph{3}.htheta = struct('y', [], 'u', []);


end

