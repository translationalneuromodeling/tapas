%% Hierarchical Unsupervised Generative Embedding
%% Introduction
% This toolbox implements Hierarchical Unsupervised Generative Embedding (HUGE) 
% with variational Bayesian inversion. This demo script provides a quick-start 
% guide to using the HUGE toolbox to stratify heterogeneous cohorts via generative 
% embedding or perform empirical Bayesian analysis. For a more detailed documentation, 
% see the HUGE manual. For more information on the theory behind HUGE, please 
% consult _Yao et al. Variational Bayesian Inversion for Hierarchical Unsupervised 
% Generative Embedding (HUGE). NeuroImage (2018)_, available at <https://doi.org/10.1016/j.neuroimage.2018.06.073 
% https://doi.org/10.1016/j.neuroimage.2018.06.073>.
%% Generating Synthetic Data
% Before applying HUGE, we need some data. Here, we use a simulated dataset, 
% which we generate using the toolbox itself. The dataset is based on a three-region 
% bilinear DCM with 20 subjects divided into two groups of 10 subjects each.
% 
% First, we define the network structure of the DCM and generate the experimental 
% stimuli.
%%
% fix random number generator seed for reproducible results
rng(8032, 'twister')

% define DCM network structure
dcm = struct( );
dcm.n = 3;
dcm.a = logical([ ...
    1 0 0; ...
    1 1 1; ...
    1 1 1; 
]);
dcm.c = false(dcm.n, 3);
dcm.c([1, 5]) = true;
dcm.b = false(dcm.n, dcm.n, 3);
dcm.b(:, :, 3) = logical([ ...
    0 0 0; ...
    1 0 1; ...
    1 1 0; ...
]);
dcm.d = false(dcm.n, dcm.n, 0);


% generate experimental stimuli
U = struct();
U.dt = 1.84/16;
tmp = tapas_huge_boxcar(U.dt, [24*13 24], [2 26], [3/4 16/26], [0 0;0 0]);
nSmp = length(tmp{1}) + 160;
tmp{1}(nSmp) = 0;
tmp{2}(nSmp) = 0;
tmp{2} = tmp{1}.*tmp{2};
tmp{3} = zeros(1, 24);
tmp{3}([2 3 1 4 3 1] + (0:5)*4) = 1;
tmp{3} = reshape(repmat(tmp{3}, round(26/U.dt), 1), [], 1);
tmp{3}(nSmp) = 0;
tmp{3} = tmp{3}.*tmp{2};
tmp{4} = zeros(1, 24);
tmp{4}([4 2 3 1 1 4] + (0:5)*4) = 1;
tmp{4} = reshape(repmat(tmp{4}, round(26/U.dt), 1), [], 1);
tmp{4}(nSmp) = 0;
tmp{4} = tmp{4}.*tmp{2};
tmp = tmp(2:4);

U.u = circshift(cell2mat(tmp), 6*16, 1);
U.name = {'1st stim', '2nd stim', '3rd stim'};
dcm.U = U;
dcm.Y.dt = 16*dcm.U.dt;
dcm.Y.name = {'1st reg'; '2nd reg'; '3rd reg'};
dcm.TE = .03;

% plot stimuli
figure;
for ip = 1:3
    subplot(3, 1, ip);
    plot(U.u(:, ip));
    ylabel(U.name{ip})
end
xlabel('sample')
%% 
% Now, we specify two subgroups within our simulated cohort, which differ 
% in their effective connectivity profile. The first group prefers a bottom-up 
% configuration (1st region -> 2nd region -> 3rd region) while the second group 
% prefers a top-down configuration (1st region -> 3rd region -> 2nd region).
%%
sigma = .141; % group standard deviation
listGroups = cell(2, 1);

% group 1 (bottom-up)
dcm.Ep.A = [-.1  .0  .0; ...
             .2 -.1  .0; ...
             .0  .1 -.1;];
dcm.Ep.B = zeros(dcm.n, dcm.n, 3);
dcm.Ep.B(2, 1, 3) = .2;
dcm.Ep.B(3, 2, 3) = .35;
dcm.Ep.C = double(dcm.c)*.5;
dcm.Ep.D = double(dcm.d);
dcm.Ep.transit = zeros(dcm.n,1);
dcm.Ep.decay = zeros(dcm.n,1);
dcm.Ep.epsilon = 0;
tmp = [dcm.a(:);dcm.b(:);dcm.c(:);dcm.d(:)];
dcm.Cp = diag([double(tmp).*sigma.^2; ones(2*dcm.n+1, 1)*exp(-6)]);
listGroups{1} = dcm;

% group 2 (top-down)
dcm.Ep.A = [-.1  .0   .0; ...
             .0 -.1   .1; ...
             .4 -.15 -.1;];
dcm.Ep.B = zeros(dcm.n, dcm.n, 3);
dcm.Ep.B(3, 1, 3) = .35;
dcm.Ep.B(2, 3, 3) = .35;
listGroups{2} = dcm;

%% 
% If you are familiar with SPM, you may have noticed that the data format 
% we use to specify our desired DCM network structure resembles the DCM structure 
% used by SPM. The HUGE toolbox uses the DCM structure of SPM to import experimental 
% data, export results or specify the network structure for simulating data.
% 
% The HUGE toolbox itself is implemented as a Matlab class. Hence, the first 
% step to using HUGE is to create an instance of the class using the _tapas_HUGE_ 
% command:
%%
hugeSim = tapas_Huge('Tag', 'simulated 3-region DCM');
%% 
% The _tapas_HUGE_ command takes optional parameters in the form of name-value 
% pair arguments. In the above example, we added a tag to the newly created HUGE 
% object, containing a short description of the object. To get a list of all available 
% name-value pairs, type:

help tapas_huge_property_names
%% 
% Now, we are ready to generate the dataset by simulating fMRI time series 
% for each subject. After simulation, the ground truth model parameters are stored 
% in the class property _model_. The fMRI time series (simulated or experimental) 
% are stored in the class property called _data_ and can be exported to SPM's 
% DCM format using the _export_ function.
%%
groupSizes = [10 10]; % simulate 10 subjects for each group
snr = .5; % using a signal-to-noise ratio of 0.5 (-3 dB)
hugeSim = hugeSim.simulate(listGroups, groupSizes, 'Snr', snr);

% plot simulated fMRI time series for one subject
n = 1;
figure;
for ip = 1:3
    subplot(3, 1, ip);
    plot(hugeSim.data(n).bold(:, ip));
    ylabel(hugeSim.labels.regions{ip})
end
xlabel('scans')

% export data to SPM's DCM format
[ listDcms ] = hugeSim.export();

%% Stratification of Heterogeneous Cohorts
% Using the simulated dataset from the previous section, we demonstrate how 
% to stratify the cohort into the bottom-up and top-down subgroups. All we have 
% to do is to call the method _estimate_.
%%
% invert the HUGE model for the current dataset
% (this may take some time)
hugeSim = hugeSim.estimate('K', 2, 'Verbose', true);
%% 
% This method accepts the same name-value pair arguments as the _tapas_Huge_ 
% command we used to create the HUGE object in the last section. In the example 
% above, we invert the HUGE model with two clusters and also activate command 
% line outputs. The result is saved in the property _posterior_ and can be plotted 
% using the method _plot_. Since, we are using simulated data for which ground 
% truth group assignments are available, the model will automatically calculate 
% the balanced purity of the estimation result.
%%
% plot result
plot(hugeSim)

% negative free energy
fprintf('Negative free energy: %e.\n', hugeSim.posterior.nfe)

% balanced accuracy
fprintf('Balanced accuracy: %f.\n', hugeSim.posterior.bPurity)

% export result to SPM's DCM format
resultK2 = hugeSim.export();

%% 
% Note how plotting the object generates two graphs. The first one shows 
% the estimated group assignments for each subject. The second one shows the posterior 
% estimates of the group-level DCM connectivity parameters for each group.
% 
% The negative free energy can be used to compare different models. For example, 
% we can invert the HUGE model with three clusters, see how the negative free 
% energy changes and calculate the Bayes factor between the two and three cluster 
% models.
%%
% invert the HUGE model with 3 clusters
% (this may take some time)
hugeSimK3 = hugeSim.estimate('K', 3);

% compare negative free energy between 2 and 3 cluster models
fprintf('Difference in negative free energy: %f.\n', hugeSim.posterior.nfe - hugeSimK3.posterior.nfe)

% calculate Bayes factor
BF_23 = exp(hugeSim.posterior.nfe - hugeSimK3.posterior.nfe);
fprintf('Bayes factor: %e.\n', BF_23)
%% 
% Note that options set via name-value pair arguments are persistent across 
% the lifetime of the object. For example, the object remembers that the option 
% _verbose_ has been set to _true_ in the previous section, and keeps generating 
% command line outputs.
% 
% The difference in the negative free energy shows that the model prefers 
% the two-cluster solution, which is what we would expect, given that the generating 
% model contained two subgroups.
%% Importing Data
% In the previous example, we used the HUGE object itself to generate data. 
% It is of course possible to import (experimental) fMRI time series into a HUGE 
% object. The most convenient method is to use either the _tapas_Huge_ or the 
% _estimate_ commands with the appropriate name-value pair argument.
%%
% select the first 10 subjects (bottom-up group) from the previous example
listDcmsG1 = listDcms(1:10);

% create a new HUGE object and import the data from these subjects
hugeEb = tapas_Huge('Tag', 'Group 1 - bottom-up', 'Dcm', listDcmsG1);

%% 
% Note that the data to be imported has to be stored in a cell array where 
% each cell holds a DCM struct in SPM's DCM format. This allows for convenient 
% transfer of data from SPM to HUGE.
%% Empirical Bayes
% Previously, we demonstrated how to use the HUGE toolbox to stratify a heterogeneous 
% cohort. However, HUGE can also be used to perform empirical Bayesian analysis; 
% i.e., to invert the DCMs while learning the prior over DCM parameters from the 
% data. This is accomplished by setting the number of clusters to one.
%%
% empirical Bayes
% (this may take some time)
hugeEb = hugeEb.estimate('K', 1, 'PriorClusterVariance', 0.02, 'Verbose', true);

% plot result
plot(hugeEb);
%% 
% Here, we also used a name-value pair argument to change the prior cluster 
% variance. Note also that the _plot_ method generates different graphs for empirical 
% Bayesian analysis than for stratification. The first graph shows a boxplot of 
% the posterior mean DCM parameter estimates of each individual subject. The second 
% plot shows the population-level posterior mean and standard deviation.
% 
% In this case the second graph reveals that for example, the connection 
% from the first to the second region in the current cohort is stronger than the 
% standard prior value of zero.
% 
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <http://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 

