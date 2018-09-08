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

state = struct('graph', [], 'llh', [], 'kernel', [], 'T', [], 'time', tic);

state.graph = cell(5, 1);
state.llh = cell(5, 1);
state.kernel = cell(5, 1);
state.T = cell(5, 1);

% First node
state.graph{1} = data;

% Second node
state.graph{2} = struct('y', [], 'u', []);
state.graph{2}.y = tapas_sem_multiv_init_theta(data, model, inference); 

% Regressors
state.graph{2}.u = struct(...
    'x', ptheta.x, ...
    'omega', ptheta.omega, ... x'x + I
    'iomega', ptheta.iomega, ... inv(x'x + I)
    'comega', ptheta.comega, ...
    'ciomega', ptheta.ciomega, ...
    'temperature_ordering', uint16(1:nc)); % sqrt(omega)

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

kernel = rmfield(inference.kernel{2}, 'nuk');
state.kernel{2}(:) = {kernel};

state.T{1} = model.graph{1}.htheta.T;

state.v = zeros(ns, nc);

state.nsample = 0;

end
