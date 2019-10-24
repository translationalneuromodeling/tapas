function [posterior] = tapas_sem_multiv_prepare_posterior(data, model, ...
    inference, states)
%% Prepate the output of the inference algorithm. Use the same method as
% the tapas_sem_hier.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

posterior = tapas_sem_hier_prepare_posterior(data, model, inference, ...
    states);

end

