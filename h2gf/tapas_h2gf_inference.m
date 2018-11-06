function [inference] = tapas_h2gf_inference(inference, pars)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~isfield(inference, 'estimate_method')
    inference.estimate_method = @tapas_mcmc_blocked_estimate;
end

if ~isfield(inference, 'initilize_states')
    inference.initilize_states = @tapas_h2gf_init_states;
end

if ~isfield(inference, 'initilize_state')
    inference.initilize_state = @tapas_h2gf_init_state;
end

if ~isfield(inference, 'sampling_methods')
    inference.sampling_methods = {
        @(d, m, i, s) tapas_mh_mc3_adaptive_ti_sample_node(d, m, i, s, 2), ... 
        @(d, m, i, s) tapas_sampler_dlinear_gibbs_node(d, m, i, s, 3), ...
        @tapas_sampler_mc3g ...  Use population mcmc step with generalized
        % Temperature for possibly Bayesian predictive distribution.
        };
end

if ~isfield(inference, 'metasampling_methods')
    inference.metasampling_methods = {@tapas_mcmc_meta_diagnostics, ...
        @tapas_mcmc_meta_adaptive, ...
        };
end

if ~isfield(inference, 'get_stored_state')
    inference.get_stored_state = @tapas_h2gf_get_stored_state;
end

if ~isfield(inference, 'prepare_posterior')
    inference.prepare_posterior = @tapas_h2gf_prepare_posterior;
end

if ~isfield(inference, 'mh_sampler')
    inference.mh_sampler = cell(4, 1);
end

if ~isfield(inference.mh_sampler{2}, 'propose_sample')
    inference.mh_sampler{2}.propose_sample = ...
        @tapas_mh_mc3_propose_gaussian_sample;
    inference.mh_sampler{2}.ar_rule = ...
        @tapas_mh_mc3g_arc;
end

% Take the parameters from pars, and overwrite what ever might be inference.
inference.niter = pars.niter;
inference.nburnin = pars.nburnin;
inference.mc3it = pars.mc3it;
inference.thinning = pars.thinning;
inference.ndiag = pars.ndiag;
inference.rng_seed = pars.rng_seed;

end
