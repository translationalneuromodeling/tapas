function [model] = tapas_sem_multiv_model(data, ptheta, pars)
%% Set up the model.
%
% Input
%       data        -- Data
%       ptheta      -- Definition of the model
%       pars        -- Parameters structure.
% Output
%       model       -- Model structre.
%       

%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

model = struct('graph', []);
model.graph = cell(4, 1);

for i = 1:4
    model.graph{i} = struct('llh', [], 'htheta', []);
end

model.graph{1}.llh = @tapas_sem_multiv_llh;
model.graph{2}.llh = @tapas_mdlinear_hier_llh;
model.graph{3}.llh = @tapas_mdlinear_llh;
model.graph{4}.llh = []; %@ tapas_mixedlinear_llh;
model.graph{5}.llh = [];

% Computes the likelihood for a single node.
model.graph{2}.llh_sn = @tapas_mdlinear_hier_llh_sn;

[~, np] = size(ptheta.jm);
[ns, nr] = size(ptheta.x);
ng = size(ptheta.mixed, 2);

% Create a zerod matrix
ptheta.sm = tapas_zeromat(ptheta.jm);

% No create the starting points
model.graph{1}.htheta = struct('pe', 0.5, 'T', pars.T, 'model', ptheta);
model.graph{2}.htheta = struct('T', ones(size(pars.T, 2)), 'y', []);
model.graph{3}.htheta = struct('T', ones(size(pars.T, 2)), 'y', []);
model.graph{4}.htheta = struct('T', ones(size(pars.T, 2)), 'y', []);
model.graph{5}.htheta = struct('T', ones(size(pars.T, 2)), 'y', []);

% This will be used to define the states.
model.graph{3}.htheta.y.mu = zeros(nr, np);
model.graph{3}.htheta.y.pe = ones(1, np);
model.graph{3}.htheta.u = struct('x', ptheta.mixed);

% Fourth level only include the means, which has the dimensions of the 
% number of groups and parameters
model.graph{4}.htheta.y.mu = zeros(nr, np);
model.graph{4}.htheta.u = [];

% The fifth level model the prior mean and variance
model.graph{5}.htheta.y.mu = zeros(ng, np);
model.graph{5}.htheta.y.pe = ones(ng, np);

model.graph{5}.htheta.u = [];

end
