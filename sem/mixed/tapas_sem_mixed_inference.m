function [inference] = tapas_sem_mixed_inference(inference, pars)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Use the same defaults.
[inference] = tapas_sem_multiv_inference(inference, pars);

% Overwrite the infernce to include the mixedlinear gibbs step.
inference.sampling_methods = {
    @(d, m, i, s) tapas_mh_mc3_tempering_sample_node(d, m, i, s, 2), ... 
    @(d, m, i, s) tapas_sampler_mdlinear_gibbs_node(d, m, i, s, 3), ...
    @(d, m, i, s) tapas_sampler_mixedlinear_gibbs_node(d, m, i, s, 4), ...
    ... Use population mcmc step with generalized temperature for 
    ... possibly Bayesian predictive distribution.
    @tapas_sampler_mc3_tempering ...
    };

inference.get_stored_state = @tapas_sem_mixed_get_stored_state;
inference.initialize_states = @tapas_sem_mixed_init_states; 
inference.initialize_state = @tapas_sem_mixed_init_state;

end
