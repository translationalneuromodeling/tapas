function [inference] = tapas_sem_multiv_inference(inference, pars)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~isfield(inference, 'estimate_method')
    inference.estimate_method = @tapas_mcmc_blocked_estimate;
end

if ~isfield(inference, 'initialize_states')
    inference.initialize_states = @tapas_sem_multiv_init_states;
end

if ~isfield(inference, 'initialize_state')
    inference.initialize_state = @tapas_sem_multiv_init_state;
end

if ~isfield(inference, 'sampling_methods')
    inference.sampling_methods = {
        @(d, m, i, s) tapas_mh_mc3_tempering_sample_node(d, m, i, s, 2), ... 
        @(d, m, i, s) tapas_sampler_mdlinear_gibbs_node(d, m, i, s, 3), ...
        @tapas_sampler_mc3_tempering
        ...  Use population mcmc step with generalized
        % Temperature for possibly Bayesian predictive distribution.
        };
end

if ~isfield(inference, 'metasampling_methods')
    inference.metasampling_methods = {@tapas_mcmc_meta_diagnostics, ...
        @tapas_mcmc_meta_adaptive, ...
%        @tapas_mcmc_meta_adaptive_ti, ...
        };
end

if ~isfield(inference, 'get_stored_state')
    inference.get_stored_state = @tapas_sem_multiv_get_stored_state;
end

if ~isfield(inference, 'prepare_posterior')
    inference.prepare_posterior = @tapas_sem_multiv_prepare_posterior;
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
        inference.thinning = 1;
    end
end

if ~isfield(inference, 'ndiag')
    if isfield(pars, 'ndiag')
        inference.ndiag = pars.ndiag;
    else
        inference.ndiag = 200;
    end
end 

% Kernel of the algorithm. It defaults to the identity matrix. If a vector
% is entered, then a matrix is used.
if ~isfield(inference, 'k')
    inference.k = 1;
end

% No update. Parameters that are not updated. It defaults to no all 
% parameters being updated
if ~isfield(inference, 'no_update')
    inference.no_update = 0;
end

% Scale of the kernel 
if ~isfield(inference, 'kernel_scale')
    inference.kernel_scale = 0.1 * 0.1;
end


end
