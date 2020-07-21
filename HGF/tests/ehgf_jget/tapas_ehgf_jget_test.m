% Scritp to test the ehgf_jget model
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2020 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% ADAPT tapas_gaussian_obs_config.m TO CONTAIN THE FOLLOWING SETTINGS:
% c.logzemu = log(10);
% c.logzesa = 5^2;

% Start from clean slate
close all; clear variables; clc;
% Add path to toolbox
addpath(genpath('../..'));
% Load trajectories of input and ground truth
load('TRT_5trajs.mat');
% Which of the five possible input sequences?
inseq = 1;
% Number of inputs
n = size(TRT_Data.rew_traj_set, 1);
% Initialize inputs
u = NaN(n, 3);
% Fill inputs
u(:,1) = TRT_Data.rew_traj_set(:, inseq);
u(:,2) = TRT_Data.mu_traj_set(:, inseq);
u(:,3) = TRT_Data.sd_traj_set(:, inseq);
% Perceptual parameters
c_prc = [u(1,1) 1 16 1 -4 -4 1 4 1 1 1 8 -1 0 4 2];
% Observation parameters
c_obs = [log(10)];
% Simulate responses
sim = tapas_simModel(u, 'tapas_ehgf_jget', c_prc, 'tapas_gaussian_obs', c_obs, 123456789);
% Plot simulation
tapas_ehgf_jget_plotTraj(sim)
% Estimate from simulated values
est = tapas_fitModel(sim.y, sim.u, 'tapas_ehgf_jget_config', 'tapas_gaussian_obs_config');
% Plot estimation
tapas_ehgf_jget_plotTraj(est)
