function [inference] = tapas_hgf_inference(inference, pars)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~isfield(inference, 'estimate_method')
    inference.estimate_method = @tapas_mcmc_blocked_estimate;
end

if ~isfield(inference, 'initialize_states')
    inference.initialize_states = @tapas_hgf_init_states;
end

if ~isfield(inference, 'initialize_state')
    inference.initialize_state = @tapas_hgf_init_state;
end

if ~isfield(inference, 'sampling_methods')
    inference.sampling_methods = {
        @(d, m, i, s) tapas_mh_mc3_sample_node(d, m, i, s, 2), ... 
        @(d, m, i, s) tapas_sampler_vlinear_gibbs_node(d, m, i, s, 3), ...
            @tapas_sampler_mc3};
end

if ~isfield(inference, 'metasampling_methods')
    inference.metasampling_methods = {@tapas_mcmc_meta_diagnostics, ...
        @tapas_mcmc_meta_adaptive};
end

if ~isfield(inference, 'get_stored_state')
    inference.get_stored_state = @tapas_vlinear_get_stored_state;
end

if ~isfield(inference, 'prepare_posterior')
    inference.prepare_posterior = @tapas_linear_prepare_posterior;
end

if ~isfield(inference, 'mh_sampler')
    inference.mh_sampler = cell(4, 1);
end

if ~isfield(inference.mh_sampler{2}, 'propose_sample')
    inference.mh_sampler{2}.propose_sample = ...
        @tapas_mh_mc3_propose_gaussian_sample;
    inference.mh_sampler{2}.ar_rule = ...
        @tapas_mh_mc3_arc;
end


if ~isfield(inference, 'niter')
    if isfield(pars, 'niter')
        inference.niter = pars.niter;
    else
        inference.niter = 5000;
    end
end

if ~isfield(inference, 'nburnin')
    if isfield(pars, 'nburnin')
        inference.nburnin = pars.nburnin;
    else
        inference.nburnin = 5000;
    end
end

if ~isfield(inference, 'mc3it')
    if isfield(pars, 'mc3it')
        inference.mc3it = pars.mc3it;
    else
        inference.mc3it = 10;
    end
end

if ~isfield(inference, 'thinning')
    if isfield(pars, 'thinning')
        inference.thinning = pars.thinning;
    else
        inference.thinning = 0;
    end
end

if ~isfield(inference, 'ndiag')
    if isfield(pars, 'ndiag')
        inference.ndiag = pars.ndiag;
    else
        inference.ndiag = 200;
    end
end 

end
