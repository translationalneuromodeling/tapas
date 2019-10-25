function [state] = tapas_h2gf_init_state(data, model, inference)
%% Generate the structure of the states of the sampler.
%
% Input
%       
% Output
%       state       -- Initial expansion point of the algorithm.
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Number of chains
nc = size(model.graph{1}.htheta.T, 2);

% Number of subjects
ns = size(data, 1);

% Prior of the parameters
mu = model.graph{1}.htheta.hgf.mu;

% Dimenions of mu
nb = numel(mu);

state = struct('graph', [], 'llh', [], 'kernel', [], 'T', []);

state.graph = cell(4, 1);
state.llh = cell(4, 1);
state.kernel = cell(4, 1);
state.T = cell(4, 1);

% Data of each subject
state.graph{1} = data;
state.graph{2} = struct('y', [], 'u', []);

% Subject specific parameters
state.graph{2}.y = cell(ns, nc);
state.graph{2}.y(:) = {mu};

% Population mean and variance
state.graph{3} = struct('y', [], 'u', []);
state.graph{3}.y = cell(1, nc);
state.graph{3}.y(:) = {struct('mu', mu, 'pe', ones(nb, 1))};

% Use the hyperpriors as state
state.graph{4} = struct('y', [], 'u', []);
state.graph{4}.y = cell(1, nc);
state.graph{4}.y(:) = {model.graph{4}.htheta.y};


% Compute the likelihood
state.llh{1} = tapas_h2gf_llh(...
    data, ...
    state.graph{2}, ... Parameters of the model. 
    model.graph{1}.htheta); % Mostly Hgf

state.llh{2} = -inf * ones(ns, nc);
state.llh{3} = -inf * ones(1, nc);

% Kernels used for the MH step, i.e., the covariance of the proposal
% distribution
state.kernel{2} = cell(ns, nc);
state.kernel{2}(:) = inference.kernel(2);

% Temperatur schedule
state.T{1} = model.graph{1}.htheta.T;

% v is an array with the samples that have been accepted. This is useful
% for diagnostics of the acceptance rate.
state.v = zeros(ns, nc);

% Current sample index.
state.nsample = 0;

end
