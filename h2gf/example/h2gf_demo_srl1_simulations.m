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

om3 = [-7.0, -6.8, -6.7, -6.6, -6.5, -6.4, -6.3, -6.2,...
       -6.1, -6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4,...
       -5.3, -5.2, -5.1, -5.0, -5.0, -4.9, -4.9, -4.9,...
       -4.8, -4.8, -4.8, -4.7, -4.7, -4.6, -4.6, -4.4];

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
                                                om3(i)],...
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

pars.niter = 1000;
%% 
% Number of samples in the burn-in phase.

pars.nburnin = 1000;
%% 
% Number of samples used for diagnostics. During burn-in the parameters 
% of the algorithm are adjusted to increase efficiency. This happens after every 
% diagnostic cycle, whose length is defined here.

pars.ndiag = 50;
%% 
% Set up the so called temperature schedule. This is used to compute the 
% model evidence. It is a matrix of dimension NxM, where N is the number of subjects 
% and M is the number of chains used to compute the model evidence. The temperature 
% schedule is selected using a 5th order power rule. 

pars.T = ones(num_subjects, 1) * linspace(0.01, 1, 8).^5;
%% 
% How often a 'swap' step is performed.

pars.mc3it = 0;
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
clear om3;
% plot simulation
for i = 1:32
ze_est(i,1)=h2gf_est.summary(i).obs_mean;
end

for i = 1:32
ze(i,1)=sim(i).p_obs.ze;
end

for i = 1:32
om2_est(i,1)=h2gf_est.summary(i).prc_mean(13,1);
end

for i = 1:32
om2(i,1)=sim(i).p_prc.p(1,13);
end

for i = 1:32
om3_est(i,1)=h2gf_est.summary(i).prc_mean(14,1);
end

for i = 1:32
om3(i,1)=sim(i).p_prc.p(1,14);
end

fig1 = figure;
fig1.Color = [1,1,1];
subplot(1,3,1); scatter(om2',om2_est,'filled');hold on;
line(xlim, [h2gf_est.hgf.c_prc.priormus(13,1), h2gf_est.hgf.c_prc.priormus(13,1)], ...
        'LineWidth', 2, 'Color', 'black');
title('om2'); hold on;
xlabel('value in')
ylabel('estimated value')

subplot(1,3,2); scatter(om3', om3_est,'filled');hold on;
line(xlim, [h2gf_est.hgf.c_prc.priormus(14,1), h2gf_est.hgf.c_prc.priormus(14,1)], ...
        'LineWidth', 2, 'Color', 'black');
title('om3'); hold on;
xlabel('value in')
ylabel('estimated value')

subplot(1,3,3); scatter(ze', ze_est,'filled');hold on;
line(xlim, [h2gf_est.hgf.c_obs.priormus, h2gf_est.hgf.c_obs.priormus], ...
    'LineWidth', 2, 'Color', 'black');
title('ze'); hold on;
xlabel('value in')
ylabel('estimated value')