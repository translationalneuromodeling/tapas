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
% ---------------------------------------------------------------------------
%
% In addition to the standard fields in the model structure, it is possible
% to specify the priors of the empirical prior in the field 
% model.empirical_prior.The mean of the population is assumed to be the prior
% mean provided in the hgf structure.
%
%   model.empirical_prior.alpha
%
%       Alpha parameter of the gamma distribution. Defaults to (eta + 1)/2
%
%   model.empirical_prior.beta
%
%       Beta parameter of the gamma distribution.
%
%   model.empirical_prior.eta
%
%       eta parameter of the Gamma distribution. This is the weighting factor
%       given to the prior. Default to 1.
%
%  -------------------------------------------------------------------------
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
% pars.initialize_states
%
%		Function used to initialize the cell array that contains the states
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
%
% --------------------------------------------------------------------------
%
% posterior.data
%
%       Input data structure.
%
% posterior.pars
%
%       stucture containing all the parameters used by the algorithm. Fields
%       not provided by the user are filled with the default values.
%
% posterior.fe
%
%       Estimated log model evidence, i.e., free energy
%
% posterior.llh
%
%       Array of the sampled log likelihood with dimensions (number of 
%       subjects) X (number of chains) X (niter/thinning)
%
% posterior.hgf
%
%       hgf structure used by the algorithm.
%
% posterior.summary
%
%       summary structure computed with tapas_h2gf_summary
%
% posterior.T
%
%       Temperature schedule used for TI or WBIC
%
% posterior.waic
%
%       Watanabe-Akaike information criterion
%
% posterior.accuracy
%
%       Estimated accuracy defined as E[log p(y|theta)]
%
% posterior.samples
%       
%       Samples of the subject specific parameters, population mean and 
%       variance
%

% aponteeduardo@gmail.com, chmathys@ethz.ch
% copyright (C) 2019-2020
%

if nargin < 3
    pars = struct();
end

% Supplement supplied parameter values with defaults
[pars] = tapas_h2gf_pars(data, model, pars);
% Set up the h2gf hierarchy
[model] = tapas_h2gf_model(data, model, pars);
% Set the function handles needed for inference
[inference] = tapas_h2gf_inference(pars);
% Run
[posterior] = tapas_h2gf_estimate_interface(data, model, inference);


end

