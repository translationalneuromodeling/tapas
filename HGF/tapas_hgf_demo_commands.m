%% This script contains the raw commands of the tutorial demo (tapas_hgfTB_demo.m) for the HGF toolbox
% ----------------------------------------------------------------------------------------------------
% Copyright (C) 2014 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

%%
u = load('example_binary_input.txt');

%%
bopars = tapas_fitModel([], u, 'tapas_hgf_binary_config', 'tapas_bayes_optimal_binary_config', 'tapas_quasinewton_optim_config');

%%
sim = tapas_simModel(u, 'tapas_hgf_binary', [NaN 0 1 NaN 1 1 NaN 0 0 NaN 1 NaN -2.5 -6], 'tapas_unitsq_sgm', 5);

%%
tapas_hgf_binary_plotTraj(sim)

%%
est = tapas_fitModel(sim.y, sim.u, 'tapas_hgf_binary_config', 'tapas_unitsq_sgm_config', 'tapas_quasinewton_optim_config');
%%
tapas_fit_plotCorr(est)
%%
disp(est.optim.Corr)
%%
disp(est.optim.Sigma)
%%
disp(est.p_prc)
%%
disp(est.p_obs)
%%
tapas_hgf_binary_plotTraj(est)
%%
disp(est.traj)

%%
est1a = tapas_fitModel(sim.y, sim.u, 'tapas_rw_binary_config', 'tapas_unitsq_sgm_config', 'tapas_quasinewton_optim_config');
%%
tapas_fit_plotCorr(est1a)
%%
tapas_rw_binary_plotTraj(est1a)

%%
usdchf = load('example_usdchf.txt');

%%
bopars2 = tapas_fitModel([], usdchf, 'tapas_hgf_config', 'tapas_bayes_optimal_config', 'tapas_quasinewton_optim_config');
%%
tapas_fit_plotCorr(bopars2)
%%
tapas_hgf_plotTraj(bopars2)

%%
sim2 = tapas_simModel(usdchf, 'tapas_hgf', [1.04 1 0.0001 0.1 0 0 1 -13  -2 1e4], 'tapas_gaussian_obs', 0.00002);
%%
tapas_hgf_plotTraj(sim2)

%%
sim2a = tapas_simModel(usdchf, 'tapas_hgf', [1.04 1 1 0.0001 0.1 0.1 0 0 0 1 1 -13  -2 -2 1e4], 'tapas_gaussian_obs', 0.00005);
%%
tapas_hgf_plotTraj(sim2a)
%%
figure
plot(sim2a.traj.wt)
xlim([1, length(sim2a.traj.wt)])
legend('1st level', '2nd level', '3rd level')
xlabel('Trading days from Jan 1, 2010')
ylabel('Weights')
title('Precision weights')

%%
est2 = tapas_fitModel(sim2.y, usdchf, 'tapas_hgf_config', 'tapas_gaussian_obs_config', 'tapas_quasinewton_optim_config');
%%
tapas_fit_plotCorr(est2)
%%
tapas_hgf_plotTraj(est2)

%%
sim2b = tapas_simModel(usdchf, 'tapas_hgf', [1.04 1 0.0001 0.1 0 0 1 -15  -2.5 1e4], 'tapas_gaussian_obs', 0.00002);
%%
tapas_hgf_plotTraj(sim2b)

%%
est2b = tapas_fitModel(sim2b.y, usdchf, 'tapas_hgf_config', 'tapas_gaussian_obs_config', 'tapas_quasinewton_optim_config');
%%
tapas_fit_plotCorr(est2b)
%%
tapas_hgf_plotTraj(est2b)

%%
bpa = tapas_bayesian_parameter_average(est2, est2b);
%%
tapas_fit_plotCorr(bpa)
%%
tapas_hgf_plotTraj(bpa)
