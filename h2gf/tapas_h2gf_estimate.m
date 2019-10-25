function [posterior] = tapas_h2gf_estimate(data, model, pars)
%% Estimates the H2GF using MCMC
%
% Input:
%
%   data        -- Data of the model. It should be a structure array with 
%                  fields y, and u of dimensions n times 1, where n is the 
%                  number of subjects.
%   model       -- Standard HGF structure object. 
%   pars        -- Parameter structure. (Optional)
%       
% Output:
%
% 	 posterior    -- Structure containing the posterior.
%
% pars is a structure with the settings for the inference algorithm. If 
% empty, default values are used.
%
% pars.model_evidence_method:
%
%		Method to compute the model evidence. Either the 'wbic' or 'ti'. The
%		later uses thermodynamic integration with the temperature schedule T.
%		(see below).
%
% pars.T
%
%		Temperature schedule for thermodynamic integration. If not provided
%		it defaults to pars.nchains with a 5th order power rule.
%
% pars.nchains
%
%		Number of chains used for TI when pars.T is not provided. When pars.T
%		is provided, pars.nchains is silently ignored.
%
% pars.nburnin
%
%		Number of iterations during the burn-in phase.
%
% pars.niter
%
%		Number of iterations of the algorithm after the burn-in phase.
%
% pars.ndiag
%
%		Number of iterations between diagnostic cycles.
%
% pars.seed
%
%		Seed of the random number generator (RNG). If pars.seed defaults to
%		zero. In that case, the 'shuffle' method of matlab's rng will be 
%		used.
%
% pars.mc3it
%
%		Markov Chain Monte Carlo Multichain Method (mc3) schedule. It species
%		the number of times that mc3 swaps are attemped per cycle. It
%		defaults to zero.
%
% pars.thinning
%
%		Thinning factor for the posterior samples. Every pars.thinning 
%		samples are stored in memory and kept for the posterior. Defaults 
%		to 1.
%
% pars.estimate_method
%
%		Main loop used in the algorithm. Defaults to 
%		@tapas_mcmc_blocked_estimate
%
% pars.initilize_states
%
%		Function used to initilize the cell array that contains the states
%		stored in memory. Defaults to @tapas_h2gf_init_states
%
% pars.sampling_methods
%
%		Cell array with functions used to draw samples from the posterior
%		distribution.
%
% pars.metasampling_methods
%
%		Cell array with diagnostic and method for adaptive mcmc.
%
% pars.get_stored_state
%
%		Funcion handle with the function used to prepare the states stored in
%		memory.
%
% pars.prepare_posterior
%
%		Function used to postprocess the states after sampling.
% 
% pars.mh_sampler
%
%		Cell array of function handles used to perform MH steps.
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%


n = 2;

n = n + 1;
if nargin < n
    pars = struct();
end

[pars] = tapas_h2gf_pars(data, model, pars);
[data] = tapas_h2gf_data(data, model, pars);
[model] = tapas_h2gf_model(data, model, pars);

% The first argument is a place holder.
[inference] = tapas_h2gf_inference(pars);

[posterior] = tapas_h2gf_estimate_interface(data, model, inference);


end

