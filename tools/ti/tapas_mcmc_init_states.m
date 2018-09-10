function [states] = tapas_mcmc_init_states(data, model, inference)
%% Initilizes the cell containing the states.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nf = numel(inference.sampling_methods);
states = cell(nf * inference.niter, 1);


end

