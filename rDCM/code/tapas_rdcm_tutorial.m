function tapas_rdcm_tutorial()
% Tutorial script to demonstrate the application of regression DCM (rDCM) on
% simulated data. In the analysis, we try to recover the data-generating 
% network architecture by pruning a fully connected network using rDCM with
% embedded sparsity constraints. Accuracy of rDCM is then assessed by means 
% of sensitivity and specificity of recovering the model structure. 
% 
% IMPORTANT: RUN THIS SCRIPT FROM WITHIN THE rDCM FOLDER
% 
%	Input:
%
%   Output: 
%
% 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2018 Translational Neuromodeling Unit
%                         Institute for Biomedical Engineering
%                         University of Zurich & ETH Zurich
%
% This file is part of the TAPAS rDCM Toolbox, which is released under the 
% terms of the GNU General Public License (GPL), version 3.0 or later. You
% can redistribute and/or modify the code under the terms of the GPL. For
% further see COPYING or <http://www.gnu.org/licenses/>.
% 
% Please note that this toolbox is in an early stage of development. Changes 
% are likely to occur in future releases.
% 
% ----------------------------------------------------------------------


% fix the random number generator
rng(2406,'twister')


% load the example DCM file
temp = load([pwd '/test/DCM_LargeScaleSmith_model1.mat']);
DCM  = temp.DCM;


% specify the options for the rDCM analysis
options.SNR     = 3;
options.TR      = 0.5;
options.p0_all  = 0.15;  % single p0 value (for computational efficiency)
options.iter    = 100;


% get time
currentTimer = tic;

% run the rDCM analysis (generates synthetic data and inverts the DCM)
output = tapas_rdcm_estimate(DCM, 's', options, 2);

% output time
toc(currentTimer)


%% visualize the results

% output and visualize the results
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,2,1)
imagesc(output.Ep.A)
title('estimated')
axis square
set(gca,'xtick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
set(gca,'ytick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
xlabel('region (from)')
ylabel('region (to)')

subplot(1,2,2)
imagesc(DCM.Tp.A)
title('true')
axis square
set(gca,'xtick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
set(gca,'ytick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
xlabel('region (from)')
ylabel('region (to)')


%% evaluate the accuracy of model architecture recovery

% get true parameters
par_true = tapas_rdcm_ep2par(DCM.Tp);
idx_true = par_true ~= 0;

% get present connections
Ip_est  = tapas_rdcm_ep2par(output.Ip);
lb      = log(1/10);
idx_Ip  = log(Ip_est./(1-Ip_est)) > lb;

% specify which parameters to test
temp2.A  = ones(size(DCM.a))-eye(size(DCM.a));
temp2.B  = zeros(size(DCM.b));
temp2.C  = zeros(size(DCM.c));
vector   = tapas_rdcm_ep2par(temp2);
vector   = vector == 1;

% evaluate TP, FP, TN, FN
true_positive   = sum((idx_Ip(vector) == 1) & (idx_true(vector) == 1));
false_positive  = sum((idx_Ip(vector) == 1) & (idx_true(vector) == 0));
true_negative   = sum((idx_Ip(vector) == 0) & (idx_true(vector) == 0));
false_negative  = sum((idx_Ip(vector) == 0) & (idx_true(vector) == 1));

% evaluate sensitivity and specifity
sensitivity = true_positive/(true_positive + false_negative);
specificity = true_negative/(true_negative + false_positive);

% output the result
fprintf('\nSummary\n')
fprintf('-------------------\n\n')
fprintf('Accuracy of model architecture recovery: \n')
fprintf('Sensitivity: %.3G - Specificity: %.3G\n',sensitivity,specificity)

end
