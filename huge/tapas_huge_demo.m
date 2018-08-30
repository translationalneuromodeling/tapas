%% TAPAS HUGE demo
%% Introduction
% This toolbox implements Hierarchical Unsupervised Generative Embedding
% (HUGE) with variational Bayesian inversion. This demo shows how to call
% HUGE on two synthetic datasets. For more information, consult REF [1].
%

%%
%
% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2018 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This toolbox is part of TAPAS, which is released under the terms of the
% GNU General Public Licence (GPL), version 3. For further details, see
% <http://www.gnu.org/licenses/>
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is in an early stage of
% development. Considerable changes are planned for future releases. For
% support please refer to:
% <https://github.com/translationalneuromodeling/tapas/issues>
%


%% Generate synthetic DCM fMRI datasets
% We generate two synthetic datasets:
% 
% # A dataset based on a three-region bilinear DCM with 80 subjects divided
%   into three groups (40, 20, 20).
% # A dataset based on a two-region linear DCM with 20 subjects divided
%   into two groups of 10 subjects each.
%
rng(8032,'twister')
tapas_huge_generate_examples( 'example_data.mat' )
load( 'example_data.mat' );

%% Analyzing first dataset
% To perform analysis with HUGE use the following interface:
%
%  [DcmResults] = tapas_huge_invert(DCM, K, priors, verbose, randomize, seed) 
%

%%
% |DCM|: cell array of DCM in SPM format
disp(DCMr3b{1})

%%
% |K|: number of clusters in the HUGE model
K = 3;

%%
% |priors|: model priors stored in a struct containing the following
%           fields: 
% 
% * |alpha|:         parameter of Dirichlet prior ($\alpha_0$ in Fig.1 of
%                    REF [1])
% * |clustersMean|:  prior mean of clusters ($m_0$ in Fig.1 of REF [1])
% * |clustersTau|:   $\tau_0$ in Fig.1 of REF [1]
% * |clustersDeg|:   degrees of freedom of inverse-Wishart prior ($\nu_0$ 
%                    in Fig.1 of REF [1]) 
% * |clustersSigma|: scale matrix of inverse-Wishart prior ($S_0$ in Fig.1
%                    of REF [1]) 
% * |hemMean|:       prior mean of hemodynamic parameters ($\mu_h$ in
%                    Fig.1 of REF [1]) 
% * |hemSigma|:      prior covariance of hemodynamic parameters($\Sigma_h$
%                    in Fig.1 of REF [1]) 
% * |noiseInvScale|: prior inverse scale of observation noise ($b_0$ in
%                    Fig.1 of REF [1]) 
% * |noiseShape|:    prior shape parameter of observation noise ($a_0$ in
%                    Fig.1 of REF [1])
%
%%
% You may use |tapas_huge_build_prior(DCM)| to generate this struct.
%
priors = tapas_huge_build_prior(DCMr3b);
disp(priors)

%%
% |verbose|: activate command line output (optional).
verbose = true;

%%
% |randomize|: randomize starting values (optional). If randomize is false
%              (default), VB inversion will start from the prior values.
randomize = true;

%%
% |seed|: seed for random number generator (optional). Use this input to
%         reproduce an earlier result.
seed = rng;

%%
% Starting the inversion:
%
currentTimer = tic;
[DcmResults] = tapas_huge_invert(DCMr3b, K, priors, verbose, randomize, seed);
toc(currentTimer)
%%
% The inference result is stored in |DcmResults.posterior|, which is a
% struct containing the following fields:
% 
% * |alpha|:               parameter of posterior over cluster weights
%                          ($\alpha_k$ in Eq.(15) of REF [1]) 
% * |softAssign|:          posterior assignment probability of subjects to
%                          clusters ($q_{nk}$ in Eq.(18) in REF [1])
% * |clustersMean|:        posterior mean of clusters ($m_k$ in Eq.(16) of
%                          REF [1]) 
% * |clustersTau|:         $\tau_k$ in Eq.(16) of REF [1]
% * |clustersDeg|:         posterior degrees of freedom ($\nu_k$ in Eq.(16)
%                          of REF [1])
% * |clustersSigma|:       posterior scale matrix ($S_k$ in Eq.(16) of
%                          REF [1])
% * |logDetClustersSigma|: log-determinant of $S_k$
% * |dcmMean|:             posterior mean of DCM parameters ($\mu_n$ in 
%                          Eq.(19) of REF [1])  
% * |dcmSigma|:            posterior covariance of hemodynamic parameters
%                          ($\Sigma_n$ in Eq.(19) of REF [1]) 
% * |logDetPostDcmSigma|:  log-determinant of $\Sigma_n$
% * |noiseInvScale|:       posterior inverse scale of observation noise
%                          ($b_{n,r}$ in Eq.(21) of REF [1]) 
% * |noiseShape|:          posterior shape parameter of observation noise
%                          ($a_{n,r}$ in Eq.(21) of REF [1])
% * |meanNoisePrecision|:  posterior mean of precision of observation noise
%                          ($\lambda_{n,r}$ in Eq.(23) of REF [1])
% * |modifiedSumSqrErr|:   $b'_{n,r}$ in Eq.(22) of REF [1]
%

%%
% The negative free energy after convergence is stored in
% |DcmResults.freeEnergy|
disp(['Negative free energy is ' num2str(DcmResults.freeEnergy) ...
    ' after ' num2str(DcmResults.nIterationsActual) ' iterations.'])

%%
% We can plot the results using the command
tapas_huge_plot(DCMr3b,DcmResults);
%%
% The assignment probabilities shown in the upper left panel of the above
% figure indicates that all but three subjects could be correctly
% classified. 
% 
% The panel in the top right shows the posterior cluster mean and 95%
% marginal credible intervals.
%
% The lower panel shows the simulated BOLD measurement (black) and 25
% samples from the posterior over (noise free) BOLD signals.
%
%


%% Analyzing the second dataset
% Performing HUGE analysis on the second dataset proceeds analogously to
% the first dataset. Differences in the DCM's structure are handled
% automatically.
% 

disp(DCMr2l{1})

% choose number of clusters for the HUGE model
K = 2;

% generate priors
priors = tapas_huge_build_prior(DCMr2l);
disp(priors)

% suppress command line output
verbose = false;

% randomize starting values
randomize = true;

% analysis
currentTimer = tic;
[DcmResultsR2l] = tapas_huge_invert(DCMr2l, K, priors, verbose, randomize);
toc(currentTimer)

disp(['Negative free energy is ' num2str(DcmResultsR2l.freeEnergy) ...
    ' after ' num2str(DcmResultsR2l.nIterationsActual) ' iterations.'])

% plot the results
tapas_huge_plot(DCMr2l,DcmResultsR2l);


%% Empirical Bayes analysis
% To perform empirical Bayes analysis, set |K| to |1|. Here, we perform
% empirical Bayesian analysis on the second dataset. For more information
% on empirical Bayes, see Fig.5 in REF [1].
%

% set K to 1 for empirical Bayes
K = 1;

% generate priors
priors = tapas_huge_build_prior(DCMr2l);

% suppress command line output
verbose = false;

% randomize starting values
randomize = true;

% analysis
currentTimer = tic;
[DcmResultsEb] = tapas_huge_invert(DCMr2l, K, priors, verbose, randomize);
toc(currentTimer)

disp(['Negative free energy is ' num2str(DcmResultsEb.freeEnergy) ...
    ' after ' num2str(DcmResultsEb.nIterationsActual) ' iterations.'])

% plot the results
tapas_huge_plot(DCMr2l,DcmResultsEb);
%%
% The above figure summarizes the result of empirical Bayes. The top panel
% shows the posterior group-level mean with 95% marginal credible
% intervals. 
%
% The lower panel shows a boxplot of first-level (i.e.: subject-level)
% MAP estimates of the DCM parameters. 
%


%% References:
%
% [1] Yao Y, Raman SS, Schiek M, Leff A, Frässle S, Stephan KE (2018).
%     Variational Bayesian Inversion for Hierarchical Unsupervised
%     Generative Embedding (HUGE). NeuroImage, 179: 604-619
%     <https://doi.org/10.1016/j.neuroimage.2018.06.073>
% 
