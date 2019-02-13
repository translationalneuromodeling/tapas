%% Hierarchical HGF (h2gf) Demo

close all; clear variables; clc;
addpath(genpath('..'));
%% Preparation of the model
% You should be familiar with the HGF Toolbox (part of TAPAS) in order to understand 
% what follows. If you're not, a good way to get started with it is to work through 
% its interactive demo in HGF/hgf_demo.mlx.
% 
% Once you have a basic understanding of how to use the HGF Toolbox, you 
% can start with the h2gf is by initializing a structure to hold the HGF model 
% definition.

hgf = struct('c_prc', [], 'c_obs', []);
%% 
% Next, we need to choose a perceptual model and the corresponding reparameteriztion 
% function.

hgf.c_prc.prc_fun = @tapas_hgf_binary;
hgf.c_prc.transp_prc_fun = @tapas_hgf_binary_transp;
%% 
% Then we do the same for the observation model.

hgf.c_obs.obs_fun = @tapas_unitsq_sgm; 
hgf.c_obs.transp_obs_fun = @tapas_unitsq_sgm_transp; 
%% 
% Now we run the config function of the perceptual model and copy the priors 
% and number of levels of the perceptual model into the hgf structure.

config = tapas_hgf_binary_config();
hgf.c_prc.priormus = config.priormus;
hgf.c_prc.priorsas = config.priorsas;
hgf.c_prc.n_levels = config.n_levels;
clear config
%% 
% We set the priors of the observational model directly.

hgf.c_obs.priormus = 0.5;
hgf.c_obs.priorsas = 1;
%% 
% All of the above is familiar from non-hierarchical applications of the 
% HGF to a single dataset as implemented in the HGF Toolbox. However, once we 
% move to the hierarchical estimation of parameters across several datasets, it 
% is useful to assign each prior a weight in terms of additional, virtual datasets. 
% This is parameterized by $\eta$ (eta), which weights the priors with respect 
% to the observed datasets. Its value corresponds to the number of virtual datasets 
% represented by the priors. For example, if we have data from ten real subjects 
% and set $\eta$ to ten, the priors will have a weight equal to that of the data. 
% By default $\eta$ is one, corresponding to one virtual dataset.
% 
% $\eta$ can be a scalar, setting the same weight across all priors; or a 
% vector, setting a particular weight for each parameter's prior. If eta is a 
% vector, its length needs to be the sum of the lengths hgf.c_prc.priormus and 
% hgf.c_obs.priormus.

hgf.empirical_priors = struct('eta', []);
hgf.empirical_priors.eta = 1;
%% Data simulation
% Now let us assume that our datasets are from 32 different subjects.

num_subjects = 32;
%% 
% For our simulations, we choose a range of values for the parameters $\omega_2$ 
% (tonic volatility at the second level) and $\zeta$ (inverse decision noise). 
% The rest of the parameters will be held constant.

om2 = [-5.0, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3, -4.2,...
       -4.1, -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4,...
       -3.3, -3.2, -3.1, -3.0, -3.0, -2.9, -2.9, -2.9,...
       -2.8, -2.8, -2.8, -2.7, -2.7, -2.6, -2.6, -2.4];

ze = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.1,...
      1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5,...
      1.6, 1.6, 1.7, 1.7, 1.7, 1.8, 1.8, 1.9,...
      1.9, 2.0, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5];
%% 
% We randomly permute the elements of $\zeta$ such that there is no systematic 
% association between high values of $\omega_2$ and high values of $\zeta$.

ze = ze(randperm(length(ze))); 
%% 
% Then we initialize structure arrays for the simulations and for the 'data' 
% argument of tapas_h2gf_estimate().

sim = struct('u', [],...
             'ign', [],...
             'c_sim', [],...
             'p_prc', [],...
             'c_prc', [],...
             'traj', [],...
             'p_obs', [],...
             'c_obs', [],...
             'y', []);

%%
data = struct('y', cell(num_subjects, 1),...
              'u', [],...
              'ign', [],...
              'irr', []);
%% 
% We load the example inputs $u$ and generate simulated data with the chosen 
% range of parameter settings

[~, u] = tapas_h2gf_load_example_data();
%%
for i = 1:num_subjects
    sim(i) = tapas_simModel(u,...
                            'tapas_hgf_binary', [NaN,...
                                                1,...
                                                1,...
                                                NaN,...
                                                1,...
                                                1,...
                                                NaN,...
                                                0,...
                                                0,...
                                                1,...
                                                1,...
                                                NaN,...
                                                om2(i),...
                                                -6],...
                         'tapas_unitsq_sgm', ze(i));
    % Simulated responses
    data(i).y = sim(i).y;
    % Experimental inputs
    data(i).u = sim(i).u;
end
clear i u om2 ze
%% Sampler configuration
% The h2gf uses sampling for inference on parameter values. To configure the 
% sampler, we first need to initialize the structure holding the parameters of 
% the inference.

pars = struct();
%% 
% We set some parameters explicitly:
% 
% Number of samples stored.

pars.niter = 500;
%% 
% Number of samples in the burn-in phase.

pars.nburnin = 500;
%% 
% Number of chains.

pars.nchains = 8;
%% 
% Parameters not explicitly defined here take the default values set in 
% tapas_h2gf_inference.m.
% 
% Before proceeding, we declutter the workspace.

clear num_subjects
%% Estimation
% Before we can run the estimation, we still need to initialize a structure 
% holding the results of the inference.

inference = struct();
%% 
% Finally, we can run the estimation.

h2gf_est = tapas_h2gf_estimate(data, hgf, inference, pars);
%% 
% We can now look at the belief trajectories of individual subjects. These 
% are the trajectories implied by the median posterior parameter values.

tapas_hgf_binary_plotTraj(h2gf_est.summary(1))
%% 
% All the plotting and diagnostic functions from the HGF Toolbox work with 
% the 'summary' substructure of the h2gf output.

tapas_fit_plotCorr(h2gf_est.summary(1))
%%
tapas_fit_plotResidualDiagnostics(h2gf_est.summary(1))