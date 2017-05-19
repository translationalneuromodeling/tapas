%% This script contains a tutorial demo for the HGF toolbox
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2014 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

fprintf('\nFirst, we will load the example binary inputs provided in the\nfile example_binary_input.txt:\n')
input('\n(press ENTER)')

fprintf('\n>> u = load(''example_binary_input.txt'');\n')
u = load('example_binary_input.txt');
input('\n(press ENTER)')

fprintf('\nNow, we will find the Bayes optimal perceptual parameters for\nthis dataset under the binary HGF model:\n')
input('\n(press ENTER)')

fprintf('\n>> bopars = tapas_fitModel([], u, ''tapas_hgf_binary_config'', ''tapas_bayes_optimal_binary_config'', ''tapas_quasinewton_optim_config'');\n')
bopars = tapas_fitModel([], u, 'tapas_hgf_binary_config', 'tapas_bayes_optimal_binary_config', 'tapas_quasinewton_optim_config');
input('(press ENTER)')

fprintf('\n\nYou can now use the optimal parameters as prior means\nby adapting tapas_hgf_binary_config.m. See the manual or the\ncomments in the file itself for details on how to do that.\n')
input('\n(press ENTER)')

fprintf('\n\nNext, we simulate a non-optimal agent''s responses:\n')
input('\n(press ENTER)')

fprintf('\n>> sim = tapas_simModel(u, ''tapas_hgf_binary'', [NaN 0 1 NaN 1 1 NaN 0 0 NaN 1 NaN -2.5 -6], ''tapas_unitsq_sgm'', 5);\n')
sim = tapas_simModel(u, 'tapas_hgf_binary', [NaN 0 1 NaN 1 1 NaN 0 0 NaN 1 NaN -2.5 -6], 'tapas_unitsq_sgm', 5);
input('\n(press ENTER)')

fprintf('\nWe visualize the simulated trajectory:\n')
input('\n(press ENTER)')

fprintf('\n>> tapas_hgf_binary_plotTraj(sim)\n')
tapas_hgf_binary_plotTraj(sim)
input('\n(press ENTER)')

fprintf('\nThe general meaning of the arguments to tapas_simModel is\nexplained in the manual and in the file tapas_simModel.m.\nThe specific meaning of each argument in this example is\nexplained in the configuration files of the perceptual model\n(tapas_hgf_binary_config.m) and of the response model\n(tapas_unitsq_sgm_config.m).\n')
input('\n(press ENTER)')

fprintf('\nNow, let''s try to recover these parameters by fitting the\ncorresponding models to the simulated data:\n')
input('\n(press ENTER)')

fprintf('\n>> est = tapas_fitModel(sim.y, sim.u, ''tapas_hgf_binary_config'', ''tapas_unitsq_sgm_config'', ''tapas_quasinewton_optim_config'');\n')
est = tapas_fitModel(sim.y, sim.u, 'tapas_hgf_binary_config', 'tapas_unitsq_sgm_config', 'tapas_quasinewton_optim_config');
input('(press ENTER)')


fprintf('\nTo check how well the parameters could be identified,\nwe''ll take a look at their posterior correlation:\n')
input('\n(press ENTER)')

fprintf('\n>> tapas_fit_plotCorr(est)\n')
tapas_fit_plotCorr(est)
input('\n(press ENTER)')

fprintf('\nYou can find the posterior correlation and covariance\nin the structure returned by the estimation:\n')
input('\n(press ENTER)')

fprintf('\n>> disp(est.optim.Corr)\n\n')
disp(est.optim.Corr)
input('\n(press ENTER)')

fprintf('\n>> disp(est.optim.Sigma)\n\n')
disp(est.optim.Sigma)
input('\n(press ENTER)')

fprintf('\nThe posterior means of the estimated as well as the\nfixed parameters can be found in est.p_prc for the\nperceptual model and in est.p_obs for the observation\nmodel:\n')
input('\n(press ENTER)')

fprintf('\n>> disp(est.p_prc)\n\n')
disp(est.p_prc)
input('\n(press ENTER)')

fprintf('\n>> disp(est.p_obs)\n\n')
disp(est.p_obs)
input('\n(press ENTER)')

fprintf('\nParameters are contained in these structures separately\nby name (e.g., om for omega) as well as jointly as a vector\np in their native space and as a vector ptrans in their\ntransformed space (i.e., the space they are estimated in).\nFor details, see the manual.\n')
input('\n(press ENTER)')

fprintf('\nNow, let''s visualize the inferred belief trajectories\nimplied by the estimated parameters:\n')
input('\n(press ENTER)')

fprintf('\n>> tapas_hgf_binary_plotTraj(est)\n')
tapas_hgf_binary_plotTraj(est)
input('\n(press ENTER)')

fprintf('\nThese trajectories can be found in est.traj:\n')
input('\n(press ENTER)')

fprintf('\n>> disp(est.traj)\n\n')
disp(est.traj)
input('\n(press ENTER)')

fprintf('\nNext, let''s try to fit the same data using a different\nperceptual model:\n')
input('\n(press ENTER)')

fprintf('\n>> est1a = tapas_fitModel(sim.y, sim.u, ''tapas_rw_binary_config'', ''tapas_unitsq_sgm_config'', ''tapas_quasinewton_optim_config'');\n')
est1a = tapas_fitModel(sim.y, sim.u, 'tapas_rw_binary_config', 'tapas_unitsq_sgm_config', 'tapas_quasinewton_optim_config');
input('(press ENTER)')

fprintf('\n>> tapas_fit_plotCorr(est1a)\n')
tapas_fit_plotCorr(est1a)
input('\n(press ENTER)')

fprintf('\n>> tapas_rw_binary_plotTraj(est1a)\n')
tapas_rw_binary_plotTraj(est1a)
input('\n(press ENTER)')

fprintf('\nThe same procedure can be applied to continuous data.\nThe file example_usdchf.txt contains the value of the\nUS dollar in Swiss francs throughout much of 2010 and 2011.\n')
input('\n(press ENTER)')

fprintf('\n>> usdchf = load(''example_usdchf.txt'');\n')
usdchf = load('example_usdchf.txt');
input('\n(press ENTER)')

fprintf('\n>> bopars2 = tapas_fitModel([], usdchf, ''tapas_hgf_config'', ''tapas_bayes_optimal_config'', ''tapas_quasinewton_optim_config'');\n')
bopars2 = tapas_fitModel([], usdchf, 'tapas_hgf_config', 'tapas_bayes_optimal_config', 'tapas_quasinewton_optim_config');
input('\n(press ENTER)')

fprintf('\n>> tapas_fit_plotCorr(bopars2)\n')
tapas_fit_plotCorr(bopars2)
input('\n(press ENTER)')

fprintf('\n>> tapas_hgf_plotTraj(bopars2)\n')
tapas_hgf_plotTraj(bopars2)
input('\n(press ENTER)')

fprintf('\n>> sim2 = tapas_simModel(usdchf, ''tapas_hgf'', [1.04 1 0.0001 0.1 0 0 1 -13  -2 1e4], ''tapas_gaussian_obs'', 0.00002);\n')
sim2 = tapas_simModel(usdchf, 'tapas_hgf', [1.04 1 0.0001 0.1 0 0 1 -13  -2 1e4], 'tapas_gaussian_obs', 0.00002);
input('\n(press ENTER)')

fprintf('\n>> tapas_hgf_plotTraj(sim2)\n')
tapas_hgf_plotTraj(sim2)
input('\n(press ENTER)')

fprintf('\nBefore proceeding to recover the parameters we''ve\nput into the simulation, let''s look at a simulation\nthat uses three levels:\n')
input('\n(press ENTER)')

fprintf('\n>> sim2a = tapas_simModel(usdchf, ''tapas_hgf'', [1.04 1 1 0.0001 0.1 0.1 0 0 0 1 1 -13  -2 -2 1e4], ''tapas_gaussian_obs'', 0.00005);\n')
sim2a = tapas_simModel(usdchf, 'tapas_hgf', [1.04 1 1 0.0001 0.1 0.1 0 0 0 1 1 -13  -2 -2 1e4], 'tapas_gaussian_obs', 0.00005);
input('\n(press ENTER)')

fprintf('\n>> tapas_hgf_plotTraj(sim2a)\n')
tapas_hgf_plotTraj(sim2a)
input('\n(press ENTER)')

fprintf('\nAs a rule of thumb, adding levels makes sense as long\nas the topmost trajectory is not flat. When estimating\nmodels, the log-model evidence can be used as a criterion\nfor whether adding levels leads to an improvement.\n')
input('\n(press ENTER)')

fprintf('\nTo show the magic of precision weighting, let''s\nplot the trajectories of the precision weights at all three\nlevels. These are the factors (i.e., weights) that are\napplied to the prediction errors to update beliefs.\nThe weights shoot up in situations of high uncertainty,\nwhere learning rates need to be increased.\n')
input('\n(press ENTER)')

figure
plot(sim2a.traj.wt)
xlim([1, length(sim2a.traj.wt)])
legend('1st level', '2nd level', '3rd level')
xlabel('Trading days from Jan 1, 2010')
ylabel('Weights')
title('Precision weights')

fprintf('\nNow, let''s again try to recover the parameters\nthat went into our simulation:\n')
input('\n(press ENTER)')

fprintf('\n>> est2 = tapas_fitModel(sim2.y, usdchf, ''tapas_hgf_config'', ''tapas_gaussian_obs_config'', ''tapas_quasinewton_optim_config'');\n')
est2 = tapas_fitModel(sim2.y, usdchf, 'tapas_hgf_config', 'tapas_gaussian_obs_config', 'tapas_quasinewton_optim_config');
input('\n(press ENTER)')

fprintf('\n>> tapas_fit_plotCorr(est2)\n')
tapas_fit_plotCorr(est2)
input('\n(press ENTER)')

fprintf('\n>> tapas_hgf_plotTraj(est2)\n')
tapas_hgf_plotTraj(est2)
input('\n(press ENTER)')

fprintf('\nIt is often useful to average parameters from several estimations, for\ninstance to compare groups of subjects. This can be achieved by using\nthe function tapas_bayesian_parameter_average(...) which takes into\naccount the covariance structure between the parameters and weights\nindividual estimates according to their precision:\n')

fprintf('\n>> sim2b = tapas_simModel(usdchf, ''tapas_hgf'', [1.04 1 0.0001 0.1 0 0 1 -15  -2.5 1e4], ''tapas_gaussian_obs'', 0.00002);\n')
sim2b = tapas_simModel(usdchf, 'tapas_hgf', [1.04 1 0.0001 0.1 0 0 1 -15 -2.5 1e4], 'tapas_gaussian_obs', 0.00002);
input('\n(press ENTER)')

fprintf('\n>> tapas_hgf_plotTraj(sim2b)\n')
tapas_hgf_plotTraj(sim2b)
input('\n(press ENTER)')

fprintf('\n>> est2b = tapas_fitModel(sim2b.y, usdchf, ''tapas_hgf_config'', ''tapas_gaussian_obs_config'', ''tapas_quasinewton_optim_config'');\n')
est2b = tapas_fitModel(sim2b.y, usdchf, 'tapas_hgf_config', 'tapas_gaussian_obs_config', 'tapas_quasinewton_optim_config');
input('\n(press ENTER)')

fprintf('\n>> tapas_fit_plotCorr(est2b)\n')
tapas_fit_plotCorr(est2b)
input('\n(press ENTER)')

fprintf('\n>> tapas_hgf_plotTraj(est2b)\n')
tapas_hgf_plotTraj(est2b)
input('\n(press ENTER)')

fprintf('\n>> bpa = tapas_bayesian_parameter_average(est2, est2b);\n')
bpa = tapas_bayesian_parameter_average(est2, est2b);
input('\n(press ENTER)')

fprintf('\n>> tapas_fit_plotCorr(bpa)\n')
tapas_fit_plotCorr(bpa)
input('\n(press ENTER)')

fprintf('\n>> tapas_hgf_plotTraj(bpa)\n')
tapas_hgf_plotTraj(bpa)
input('\n(press ENTER)')

fprintf('\nNote that Bayesian parameter averaging only works for estimates that\nare based on the same priors and should only be used with care for\nestimates based on different inputs.\n')

input('\nEnd of demo - press ENTER to finish')

