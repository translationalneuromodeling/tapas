# H2GF: Hierarchical inference for the HGF
The h2gf package is an extension of the HGF for hierarchical inference in
group studies. It provides a very simple method to pool information from 
a sample population to estimate the prior mean over subjects using 
hierarchical Bayes. In addition, it can be used to compute the model
evidence using thermodynamic integration.

## Quick start
From the matlab command line, write
```matlab
% Initilize tapas
tapas_init(); 
% Run an example script
posterior = tapas_h2g_example();
```

### The example script
The h2gf package works out of the box with HGF models. The three main
inputs of the model are subjects' data, an hgf structure defining the
model, and a structure defining the parameters of the sampler.

The first part of this function sets up the model 
```matlab
% Initilize a structure to hold the hgf
hgf = struct('c_prc', [], 'c_obs', []);
% Set up the number of levels
hgf.c_prc.n_levels = 3; 

% Set up the perceptual function
hgf.c_prc.prc_fun = @tapas_hgf_binary;
% Set up the reparametrization function
hgf.c_prc.transp_prc_fun = @tapas_hgf_binary_transp;

% Set up the observation function.
hgf.c_obs.obs_fun = @tapas_unitsq_sgm; 
% Reparametrization function 
hgf.c_obs.transp_obs_fun = @tapas_unitsq_sgm_transp; 

% Enter the configuration of the binary hgf
config = tapas_hgf_binary_config();

% Priors of the perceptual model 
hgf.c_prc.priormus = config.priormus;
hgf.c_prc.priorsas = config.priorsas;

% Priors of the observational model
hgf.c_obs.priormus = 0.5;
hgf.c_obs.priorsas = 1;

% Set the empirical prior
% Eta weights the prior with respect to the observations. Because the prior
% mean is treated as fixed obserations, eta is the number of observation
% represented by the prior. If eta = 1, the prior is mean is treated as 
% a single additional observation.
hgf.empirical_priors = struct('eta', []);
hgf.empirical_priors.eta = 1;
```


Following the definition of the model, we input the data input. It is
entered as the `data` structure array
with fields `y`, `u`, `ign`, and `irr`. `y` corresponds to subjects 
responses and `u` as experimental inputs. Each row in 'data' corresponds
to a different subject.
```matlab
%% Simulating data
% Number of subjects
num_subjects = 10;

% Values of ze, ka
ze = 0.5;
ka = 0.5;

% Used if data is simulated from fixed parameters
pars = [NaN 1 1 NaN 1 1 NaN 0 0 NaN ka NaN -4 log(0.0025)];
% Initialize a structure for the data
data = struct('y', cell(num_subjects, 1), ...
    'u', cell(num_subjects, 1), 'ign', [], 'irr', []);

% These are trials used in the experiment
[y, u] = tapas_h2gf_load_example_data();

for i = 1:num_subjects
	% Generate artifical data
    %sim = tapas_simModel(u, 'tapas_hgf_binary', ...
	%	pars, 'tapas_unitsq_sgm');
    % Instead of simulating data we use the loaded data 10 time. It is also
    % possible to generate data using the function above.
	% Fill the responses
    data(i).y = y;
	% and experimental manipulations
    data(i).u = u;
end
```

At this point we enter the parameters for the inference method. It is 
a Markov Chain Monte Carlo Method that draws samples from the posterior.
Default values are set in `tapas_h2gf_inference.m`

```matlab
%% Parameters for inference
% Initilize the place holder for the parameters of the 
% inference. Missing parameters are filled by default
% values. This is implemented in tapas_h2gf_inference.m

inference = struct();
pars = struct();

% Number of samples stored 
pars.niter = 4000;
% Number of samples in the burn-in phase
pars.nburnin = 4000;
% Number of samples used for diagnostics. During the 
% burn-in phase the parameters of the algorithm are 
% adjusted to increase the efficiency. This happens after 
% every diagnostic cycle.
pars.ndiag = 1000;

% Set up the so called temperature schedule. This is used to
% compute the model evidence. It is a matrix of NxM, where N 
% is the number of subjects and M is the number of chains used 
% to compute the model evidence. The
% temperature schedule is selected using a 5th order power rule. 
pars.T = ones(num_subjects, 1) * linspace(0.01, 1, 16).^5;

% This controls how often a 'swap' step is perform. 
pars.mc3it = 0;
```

The next line runs the algorithm using the input.

```matlab
%% Run the inference method
% This function is entry point to the algorithm. Note that its
% behavior can be largely modified by changing the default 
% settings.
hgf_est = tapas_h2gf_estimate(data, hgf, inference, pars);
```


## The model
The main model used here is a 'Gamma-Gaussian' prior over the parameters
of the model. It assumes that the parameters of the model are Gaussian
distributed around the population mean. The prior of the model parameters
is effectively treated as a fixed number of observations.


























aponteeduardo@gmail.com
copyright (c) 2018
