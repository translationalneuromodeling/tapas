%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% BREATHING FILTER DETECTION TASK ANALYSIS %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% -------------------------------------------------------------------------
% Author: Olivia Harrison
% Created: 14/08/2018
%
% This software is free software: you can redistribute it and/or modify it 
% under the terms of the GNU General Public License as published by the 
% Free Software Foundation, either version 3 of the License, or (at your 
% option) any later version. This software is distributed in the hope that 
% it will be useful, but WITHOUT ANY WARRANTY; without even the implied 
% warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
% GNU General Public License for more details: http://www.gnu.org/licenses/
% -------------------------------------------------------------------------

% This analysis script uses either Brian Maniscalco's single subect
% analysis or Steve Fleming's Hierarchical Bayesian toolbox (for group
% analysis) to calculate perceptual decision and metacognitive metrics, 
% specific to data produced by running the tapas_filter_detection_task.

% IMPORTANT ANALYSIS NOTES:
%   1) The code requires JAGS software to run, which can be found here:
%      http://mcmc-jags.sourceforge.net/
%      The code has been tested on JAGS 3.4.0; there are compatibility
%      issues between matjags and JAGS 4.X.
%   2) The code requires the HMeta-d toolbox (provided as a submodule
%      from https://github.com/metacoglab/HMeta-d) to be on your path.
%   3) This script is looking for specific results files that were output
%      from the tapas_filter_detection_task, which are located in the
%      FDT/results/ folder. The results are expected to contain a 
%      results.filterThreshold.xx structure, where values for xx:
%         - results.filterThreshold.filterNum: a value to determine the
%           number of filters where trials were performed
%         - results.filterThreshold.filters: a vector with 1 meaning
%           filters were presented, 0 when filters were absent (dummy)
%         - results.filterThreshold.response: a vector with 1 meaning
%           response was 'yes', 0 when response was 'no'
%         - results.filterThreshold.confidence: a vector containing
%           confidence score (1-10) on each trial
%      If your results are formatted differently, refer directly to the
%      original scripts from the HMeta-d toolbox 
%      (https://github.com/metacoglab/HMeta-d).

% TO RUN THIS SCRIPT:
% type tapas_filter_detection_analysis into the MATLAB terminal from the 
% main FDT folder, and follow the prompts.

% ANALYSIS OPTIONS:
% This script allows you to run either:
%    - A single subject analysis: A non-Bayesian estimation of meta-d',
%      fitting one subject at a time using a single point maximum
%      likelihood estimation (from the original Maniscalco & Lau 2012
%      paper, more info can be found here: 
%      http://www.columbia.edu/~bsm2105/type2sdt/archive/index.html).
%      HOWEVER: Simulations have shown this analysis to be VERY unstable
%      when estimating meta-d' using 60 trials. It is STRONGLY encouraged
%      to utilise the hierarchical models, or collect many more trials 
%      (200+ trials) for each subject to have a more reliable measure of
%      meta-d'.
%    - A group mean analysis: A hierarchical Bayesian analysis, whereby all
%      subjects are fit together and information from the group mean is
%      used as prior information for each subject. This hierarchical model
%      helps with much more accurate estimations of meta-d' with small
%      trial numbers, such as using 60 trials per subject.
%    - A group regression analysis: A hierarchical Bayesian regression
%      analysis that is an extension on the group mean analysis, whereby 
%      subjects are fit together in one model, and the relationship between
%      single subject log(meta-d'/d') values and the covariate(s) is
%      estimated within the hierarchical model structure.
%    - A group difference analysis: A hierarchical Bayesian analysis,
%      whereby each group of subjects is fitted separately, and the groups
%      are then compared. Frequentist statistics (such as parametric
%      unpaired T-tests, or non-parametric Wilcoxon signed-rank tests) can
%      be used for all values that are not fitted using hierarchical
%      information, such as d', c, filter number, accuracy and average
%      confidence. As the group values for log(meta-d'/d') are calculated
%      using two separate hierarchical models, group differences are then
%      inferred via comparison of the resulting log(meta-d'/d')
%      distributions, and whether the 95% highest-density interval (HDI) of
%      the difference between the distributions spans zero. The HDI is the
%      shortest possible interval containing 95% of the MCMC samples, and
%      may not be symmetric. NOTE: If single subject values for
%      log(meta-d'/d') (or any related meta-d' metric) are required for
%      further analyses that span both groups (such as entry into general 
%      linear models), it is recommended to fit all subjects together in
%      one regression model with a regressor denoting group identity.
%    - A session difference analysis (paired group difference): A
%      hierarchical Bayesian analysis, whereby two sessions / measures from
%      the same participants are fitted in a single model using a
%      multivariate normal distribution. This distribution allows for the
%      non-independence between sessions for each participant. NOTE:
%      Participants must be listed in the same order for analysis, and must
%      have data for both sessions / each measure.

% NOTES ON NON-BAYESIAN SINGLE SUBJECT ANALYSIS FROM AUTHORS' WEBSITE 
% (columbia.edu/~bsm2105/type2sdt/archive/index.html):
% This analysis is intended to quantify metacognitive sensitivity (i.e. the
% efficacy with which confidence ratings discriminate between correct and
% incorrect judgments) in a signal detection theory framework. A central
% idea is that primary task performance can influence metacognitive
% sensitivity, and it is informative to take this influence into account.
% Description of the methodology can be found here: Maniscalco, B., & Lau, 
% H. (2012). A signal detection theoretic approach for estimating
% metacognitive sensitivity from confidence ratings. Consciousness and
% Cognition, 21(1), 422–430. doi:10.1016/j.concog.2011.09.021 
% If you use these analysis files, please reference the Consciousness & 
% Cognition paper and website. Brian Maniscalco: brian@psych.columbia.edu

% NOTES ON HIERARCHICAL TOOLBOX FROM ORIGINAL SCRIPTS (FOR GROUP ANALYSES):
% This MATLAB toolbox implements the meta-d’ model (Maniscalco & Lau, 2012)
% in a hierarchical Bayesian framework using Matlab and JAGS, a program for
% conducting MCMC inference on arbitrary Bayesian models. A paper with more
% details on the method and the advantages of estimating meta-d’ in a
% hierarchal Bayesian framework is available here https://academic.oup.com/
% nc/article/doi/10.1093/nc/nix007/3748261/HMeta-d-hierarchical-Bayesian-
% estimation-of. For a more general introduction to Bayesian models of 
% cognition see Lee & Wagenmakers, Bayesian Cognitive Modeling: A Practical
% Course http://bayesmodels.com/. This code is being released with a
% permissive open-source license. You should feel free to use or adapt the
% utility code as long as you follow the terms of the license provided. If
% you use the toolbox in a publication we ask that you cite the following
% paper: Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of
% metacognitive efficiency from confidence ratings, Neuroscience of
% Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007. Copyright
% (c) 2017, Stephen Fleming. For more information and/or licences, please 
% see the original code: https://github.com/metacoglab/HMeta-d.

% KEY ANALYSIS OUTPUT VALUES:
% The primary measures to come out of this analysis can be found in
% analysis.{single/groupMean/groupDiff}.xx, and they are:
%   1) xx = filterNum: Number of filters. Less filters means more sensitive
%           to changes in breathing.
%   2) xx = d1: d prime, discriminability between filter and dummy trials.
%           Larger d' means more discriminability at specific filter number
%   3) xx = c1: Decision criterion, or where the decision boundary exists.
%           Negative criterion values mean bias towards 'yes' (lower, more 
%           liberal criterion), while positive values mean bias towards 
%           'no' response (higher, more strngent criterion).
%   4) xx = meta_d: A measure of metacognition, or 'type 2' sensitivity.
%           This reflects how much information, in signal-to-noise units,
%           is available for metacognition (see Maniscalco & Lau, 2012).
%           NB: ISSUES WITH THIS ESTIMATION (AND ALL RELATED ESTIMATIONS)
%           IF USING SINGLE SUBJECT ANALYSES, AS DESCRIBED ABOVE.
%   5) xx = Mratio: Meta-d' to d' ratio, as a measure of metacognitive
%           'efficiency' (see Rouault et al., 2018).
%   6) xx = log_Mratio: log(meta_d/d1), reported in papers to help with
%           normalisation of data (see Rouault et al., 2018).
%   7) xx = avgConfidence: A second measure of metacognition, thought to be
%           independent from meta-d'. NB: Average confidence should only be
%           compared between two groups if there is no difference in d'
%           between the groups (i.e. task difficulty was comparable).

% ADDITIONAL GROUP DIFFERENCE OUTPUT VALUES:
% If the analysis is a two-group or two-session (paired) difference, the 
% following will also be calculated for the non-hierarchical measures 
% (test = groupDiff or sessionDiff; xx = filterNum, d1, c1 and 
% avgConfidence):
%   1) analysis.test.xx.h = Results of null hypothesis test, where h =
%      1 for rejection of the null, and h = 0 for no rejection.
%   2) analysis.test.xx.p = The p-value for the statistical test.
%   3) analysis.test.xx.stats = Further statistics (such as tstats,
%      number of samples, degrees of freedom) associated with the test.
%   4) analysis.test.xx.ci = Confidence interval of the difference,
%      calculated only when T-test is specified.
% The highest density interval (HDI) for the difference in log(meta-d'/d') 
% between the groups / sessions will be calculated and recorded, as 
% frequentist statistics cannot be used here. A summary Figure for each of 
% these metrics will be created and saved in the FDT/analysis/ folder.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SET UP THE ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function tapas_filter_detection_analysis()

% Check folder location is main FDT folder
[~,dir_name] = fileparts(pwd);
if ~strcmp(dir_name,'FDT')
   error('Not currently in main FDT folder. Please move to FDT folder and try again.');
end

% Add relevant paths
addpath('analysis');

% Display setup on screen
fprintf('\n________________________________________\n\n    SET UP ANALYSIS FOR FILTER TASK\n________________________________________\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHOOSE THE TYPE OF ANALYSIS TO RUN AND SPECIFY FILES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try
    analysis.type = input('Type of analysis (single or group) = ', 's'); % Ask for type of analysis to be run
catch
    fprintf('\nOption does not exist, please try again.\n');
    return
end

try
    if strcmp(analysis.type,'single') == 1 % If single subject analysis specified
        analysis.PPIDs = input('PPID = ', 's'); % Ask for PPID
        try
            analysis.data = load(fullfile('results', ['filter_task_results_', analysis.PPIDs, '.mat'])); % Load data
            analysis.ratings = analysis.data.results.setup.confidenceUpper - analysis.data.results.setup.confidenceLower + 1; % Calculate number of confidence rating bins
        catch
            fprintf('\nInvalid PPID.\n');
            return
        end
    elseif strcmp(analysis.type,'group') == 1 % If group analysis specified
        analysis.type = input('Type of analysis (mean, diff, paired or regress) = ', 's'); % Ask for type of group analysis to be run
        if strcmp(analysis.type,'mean') == 1 % If group mean analysis specified
            analysis.PPIDs = input('Input group PPIDs (e.g. {''001'', ''002'', ''003'',...}) = '); % Ask for PPIDs
            analysis.groupsize(1) = length(analysis.PPIDs);
            for n = 1:length(analysis.PPIDs)
                PPID = char(analysis.PPIDs(n));
                try
                    analysis.data(n) = load(fullfile('results', ['filter_task_results_', PPID, '.mat'])); % Load data
                catch
                    fprintf('\nInvalid PPIDs.\n');
                    return
                end
            end
            analysis.ratings = analysis.data(1).results.setup.confidenceUpper - analysis.data(1).results.setup.confidenceLower + 1; % Calculate number of confidence rating bins
        elseif strcmp(analysis.type,'diff') == 1 % If two group difference analysis specified
            analysis.PPIDs.group1 = input('Input group 1 PPIDs (e.g. {''001'', ''002'', ''003'',...}) = '); % Ask for PPIDs
            analysis.PPIDs.group2 = input('Input group 2 PPIDs (e.g. {''004'', ''005'', ''006'',...}) = '); % Ask for PPIDs
            analysis.groupDiff.test = input('Type 1 variable tests (ttest or wilcoxon) = ', 's'); % Ask for parametric or non-parmetric test options for type 1 variables (d', c etc.)
            analysis.groupsize(1) = length(analysis.PPIDs.group1);
            analysis.groupsize(2) = length(analysis.PPIDs.group2);
            for n = 1:analysis.groupsize(1)
                PPID = char(analysis.PPIDs.group1(n));
                try
                    analysis.group1.data(n) = load(fullfile('results', ['filter_task_results_', PPID, '.mat'])); % Load data
                catch
                    fprintf('\nInvalid PPIDs for group 1.\n');
                    return
                end
            end
            for n = 1:analysis.groupsize(2)
                PPID = char(analysis.PPIDs.group2(n));
                try
                    analysis.group2.data(n) = load(fullfile('results', ['filter_task_results_', PPID, '.mat'])); % Load data
                catch
                    fprintf('\nInvalid PPIDs for group 2.\n');
                    return
                end
            end
            analysis.ratings = analysis.group1.data(1).results.setup.confidenceUpper - analysis.group1.data(1).results.setup.confidenceLower + 1; % Calculate number of confidence rating bins
        elseif strcmp(analysis.type,'paired') == 1 % If paired difference analysis specified
            fprintf('\nNOTE: PPIDs from each session need to be paired and in the same order\n');
            analysis.PPIDs.session1 = input('Input session 1 PPIDs (e.g. {''001a'', ''002a'', ''003a'',...}) = '); % Ask for PPIDs
            analysis.PPIDs.session2 = input('Input session 2 PPIDs (e.g. {''001b'', ''002b'', ''003b'',...}) = '); % Ask for PPIDs
            analysis.sessionDiff.test = input('Type 1 variable tests (ttest or wilcoxon) = ', 's'); % Ask for parametric or non-parmetric test options for type 1 variables (d', c etc.)
            analysis.sessionSize(1) = length(analysis.PPIDs.session1);
            analysis.sessionSize(2) = length(analysis.PPIDs.session2);
            if analysis.sessionSize(1) ~= analysis.sessionSize(2)
                error('Number of PPIDs in each session does not match!');
            end
            for n = 1:analysis.sessionSize(1)
                PPID = char(analysis.PPIDs.session1(n));
                try
                    analysis.session1.data(n) = load(fullfile('results', ['filter_task_results_', PPID, '.mat'])); % Load data
                catch
                    fprintf('\nInvalid PPIDs for session 1.\n');
                    return
                end
            end
            for n = 1:analysis.sessionSize(2)
                PPID = char(analysis.PPIDs.session2(n));
                try
                    analysis.session2.data(n) = load(fullfile('results', ['filter_task_results_', PPID, '.mat'])); % Load data
                catch
                    fprintf('\nInvalid PPIDs for session 2.\n');
                    return
                end
            end
            analysis.ratings = analysis.session1.data(1).results.setup.confidenceUpper - analysis.session1.data(1).results.setup.confidenceLower + 1; % Calculate number of confidence rating bins
        elseif strcmp(analysis.type,'regress') == 1 % If regression analysis specified
            analysis.PPIDs = input('Input group PPIDs (e.g. {''001'', ''002'', ''003'',...}) = '); % Ask for PPIDs
            analysis.groupsize(1) = length(analysis.PPIDs);
            for n = 1:length(analysis.PPIDs)
                PPID = char(analysis.PPIDs(n));
                try
                    analysis.data(n) = load(fullfile('results', ['filter_task_results_', PPID, '.mat'])); % Load data
                catch
                    fprintf('\nInvalid PPIDs.\n');
                    return
                end
            end
            analysis.ratings = analysis.data(1).results.setup.confidenceUpper - analysis.data(1).results.setup.confidenceLower + 1; % Calculate number of confidence rating bins
            fprintf('\nNote: Covariate text file required to be placed in results folder\n--> Scores need to be in the SAME ORDER as PPID input order.\n');
            analysis.covariate.fileName = input('Input covariate file name (e.g. covariate_example.txt) = ', 's'); % Ask for covariate file name
            try
                analysis.covariate.data = load(fullfile('results', analysis.covariate.fileName)); % Load data
            catch
                fprintf('\nCovariate file cannot be found in results folder.\n');
                return
            end
            [a,b] = size(analysis.covariate.data);
            if a > b % Re-shape vector if needed
                analysis.covariate.data = analysis.covariate.data';
            end
        end
    end
catch
    fprintf('\nInvalid.\n');
    return
end

% Save results
if strcmp(analysis.type,'single') == 1
    resultsFile = fullfile('analysis', ['filter_task_analysis_', analysis.type, analysis.PPIDs]); % Create figure file name
else
    resultsFile = fullfile('analysis', ['filter_task_analysis_', analysis.type]); % Create figure file name
end
save(resultsFile, 'analysis'); % Save results


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN TRIALS2COUNTS SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(analysis.type,'single') == 1 || strcmp(analysis.type,'mean') == 1 || strcmp(analysis.type,'regress') == 1
    
    % Run trials2counts script for single subject or whole group
    for n = 1:length(analysis.data)
        stimID = analysis.data(n).results.thresholdTrials.filters;
        response = analysis.data(n).results.thresholdTrials.response;
        rating = analysis.data(n).results.thresholdTrials.confidence;
        [analysis.trials2counts.nR_S1{n}, analysis.trials2counts.nR_S2{n}] = trials2counts(stimID,response,rating,analysis.ratings,0); % Run for number of ratings and no cell padding
    end
    
elseif strcmp(analysis.type,'diff') == 1
    
    % Run trials2counts script for group 1
    for n = 1:length(analysis.group1.data)
        stimID = analysis.group1.data(n).results.thresholdTrials.filters;
        response = analysis.group1.data(n).results.thresholdTrials.response;
        rating = analysis.group1.data(n).results.thresholdTrials.confidence;
        [analysis.group1.trials2counts.nR_S1{n}, analysis.group1.trials2counts.nR_S2{n}] = trials2counts(stimID,response,rating,analysis.ratings,0); % Run for number of ratings and no cell padding
    end
    % Run trials2counts script for group 2
    for n = 1:length(analysis.group2.data)
        stimID = analysis.group2.data(n).results.thresholdTrials.filters;
        response = analysis.group2.data(n).results.thresholdTrials.response;
        rating = analysis.group2.data(n).results.thresholdTrials.confidence;
        [analysis.group2.trials2counts.nR_S1{n}, analysis.group2.trials2counts.nR_S2{n}] = trials2counts(stimID,response,rating,analysis.ratings,0); % Run for number of ratings and no cell padding
    end
    
elseif strcmp(analysis.type,'paired') == 1
    
    % Run trials2counts script for session 1
    for n = 1:length(analysis.session1.data)
        stimID = analysis.session1.data(n).results.thresholdTrials.filters;
        response = analysis.session1.data(n).results.thresholdTrials.response;
        rating = analysis.session1.data(n).results.thresholdTrials.confidence;
        [analysis.trials2counts.nR_S1(1).counts{n}, analysis.trials2counts.nR_S2(1).counts{n}] = trials2counts(stimID,response,rating,analysis.ratings,0); % Run for number of ratings and no cell padding
    end
    % Run trials2counts script for session 2
    for n = 1:length(analysis.session2.data)
        stimID = analysis.session2.data(n).results.thresholdTrials.filters;
        response = analysis.session2.data(n).results.thresholdTrials.response;
        rating = analysis.session2.data(n).results.thresholdTrials.confidence;
        [analysis.trials2counts.nR_S1(2).counts{n}, analysis.trials2counts.nR_S2(2).counts{n}] = trials2counts(stimID,response,rating,analysis.ratings,0); % Run for number of ratings and no cell padding
    end
    
end

% Save results
save(resultsFile, 'analysis');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN SINGLE SUBJECT ANALYSIS IF SPECIFIED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(analysis.type,'single') == 1 % If single subject analysis specified
    
    % Pad ratings so that zero counts do not interfere with fit
    analysis.trials2counts.nR_S1_padded = analysis.trials2counts.nR_S1{1} + 1/(2*10);
    analysis.trials2counts.nR_S2_padded = analysis.trials2counts.nR_S2{1} + 1/(2*10);
    
    % Fit data
    analysis.single.fit = fit_meta_d_MLE(analysis.trials2counts.nR_S1_padded, analysis.trials2counts.nR_S2_padded);
    
    % Pull out key variables
    analysis.single.filterNum = mean(analysis.data.results.thresholdTrials.filterNum);
    analysis.single.accuracy = analysis.data.results.thresholdTrials.accuracyTotal;
    analysis.single.d1 = analysis.single.fit.d1;
    analysis.single.c1 = analysis.single.fit.c1;
    analysis.single.meta_d = analysis.single.fit.meta_d;
    analysis.single.Mratio = analysis.single.fit.M_ratio;
    analysis.single.fit.log_Mratio = log(analysis.single.fit.M_ratio);
    analysis.single.avgConfidence = mean(analysis.data(1).results.thresholdTrials.confidence);
    
    % Save results
    save(resultsFile, 'analysis'); % Save results
    
    % Display completion message on screen
    fprintf('\n________________________________________\n\n   COMPLETED SINGLE SUBJECT ANALYSIS\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    fprintf('\n             MRATIO = %.2f\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n', analysis.single.Mratio);
    fprintf('\n        RESULTS CAN BE FOUND IN: \n             FDT/analysis/\n________________________________________\n');
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN GROUP MEAN ANALYSIS IF SPECIFIED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(analysis.type,'mean') == 1 % If group mean analysis specified
    
    % Specify parameters
    analysis.groupMean.mcmc_params = fit_meta_d_params;

    % Fit group data all at once
    analysis.groupMean.fit = fit_meta_d_mcmc_group(analysis.trials2counts.nR_S1, analysis.trials2counts.nR_S2, analysis.groupMean.mcmc_params);
    
    % Calculate log of Mratio for single subjects values
    for n = 1:length(analysis.groupMean.fit.Mratio)
        analysis.groupMean.log_Mratio.singleSubject(n) = log(analysis.groupMean.fit.Mratio(n));
    end
    
    % Pull out group mean values
    analysis.groupMean.d1.groupMean = mean(analysis.groupMean.fit.d1);
    analysis.groupMean.d1.singleSubject = analysis.groupMean.fit.d1;
    analysis.groupMean.c1.groupMean = mean(analysis.groupMean.fit.c1);
    analysis.groupMean.c1.singleSubject = analysis.groupMean.fit.c1;
    analysis.groupMean.meta_d.groupMean = mean(analysis.groupMean.fit.meta_d);
    analysis.groupMean.meta_d.singleSubject = analysis.groupMean.fit.meta_d;
    analysis.groupMean.Mratio.groupMean = exp(analysis.groupMean.fit.mu_logMratio);
    analysis.groupMean.Mratio.singleSubject = analysis.groupMean.fit.Mratio;
    analysis.groupMean.Mratio.hdi = calc_HDI(exp(analysis.groupMean.fit.mcmc.samples.mu_logMratio(:)));
    analysis.groupMean.log_Mratio.groupMean = analysis.groupMean.fit.mu_logMratio;
    analysis.groupMean.log_Mratio.hdi = calc_HDI(analysis.groupMean.fit.mcmc.samples.mu_logMratio(:));
    
    % Calculate average confidence, add filter number and accuracy to analysis results
    for n = 1:length(analysis.data)
        analysis.groupMean.avgConfidence.singleSubject(n) = mean(analysis.data(n).results.thresholdTrials.confidence);
        analysis.groupMean.filterNum.singleSubject(n) = mean(analysis.data(n).results.thresholdTrials.filterNum);
        analysis.groupMean.accuracy.singleSubject(n) = analysis.data(n).results.thresholdTrials.accuracyTotal;
    end
    analysis.groupMean.avgConfidence.groupMean = mean(analysis.groupMean.avgConfidence.singleSubject);
    analysis.groupMean.filterNum.groupMean = mean(analysis.groupMean.filterNum.singleSubject);
    analysis.groupMean.accuracy.groupMean = mean(analysis.groupMean.accuracy.singleSubject);
    
    % Save results
    save(resultsFile, 'analysis'); % Save results
    
    % Display completion message on screen
    fprintf('\n________________________________________\n\n     COMPLETED GROUP MEAN ANALYSIS\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    fprintf('\n GROUP MRATIO = %.2f (HDI: %.2f to %.2f)\n', analysis.groupMean.Mratio.groupMean, analysis.groupMean.Mratio.hdi(1), analysis.groupMean.Mratio.hdi(2));
    if (analysis.groupMean.Mratio.hdi(1) > 0 && analysis.groupMean.Mratio.hdi(2) > 0) || (analysis.groupMean.Mratio.hdi(1) < 0 && analysis.groupMean.Mratio.hdi(2) < 0)
        fprintf('   HDI DOES NOT SPAN ZERO: MRATIO SIG.\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    else
        fprintf('   HDI SPANS ZERO: MRATIO NOT SIG.\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    end
    fprintf('\n        RESULTS CAN BE FOUND IN: \n             FDT/analysis/\n________________________________________\n');
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN GROUP DIFFERENCE ANALYSIS (UNPAIRED) IF SPECIFIED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(analysis.type,'diff') == 1 % If group difference analysis specified
    
    % For two-group difference analysis: Print message about groups being fit separately --> no t-tests possible on meta-d' metrics
    fprintf('\nNOTE: Groups are fit using separate hierarchical models.\nFrequentist statistics (such as t-tests) are not possible on any metrics involving meta-d''.\nFrequentist statistics will still be applied to all type 1 metrics and average confidence values.\n');
    
    % Specify parameters
    analysis.groupDiff.mcmc_params = fit_meta_d_params;
    
    % Fit each group using a separate hierarchical Bayesian model
    analysis.group1.fit = fit_meta_d_mcmc_group(analysis.group1.trials2counts.nR_S1, analysis.group1.trials2counts.nR_S2, analysis.groupDiff.mcmc_params);
    analysis.group2.fit = fit_meta_d_mcmc_group(analysis.group2.trials2counts.nR_S1, analysis.group2.trials2counts.nR_S2, analysis.groupDiff.mcmc_params);
    
    % Compute HDI of difference for log(meta-d'/d')
    analysis.group1.logMratio.groupMean = analysis.group1.fit.mu_logMratio;
    analysis.group1.logMratio.singleSubject = log(analysis.group1.fit.Mratio);
    analysis.group1.Mratio.groupMean = exp(analysis.group1.fit.mu_logMratio);
    analysis.group1.Mratio.singleSubject = analysis.group1.fit.Mratio;
    analysis.group2.logMratio.groupMean = analysis.group2.fit.mu_logMratio;
    analysis.group2.logMratio.singleSubject = log(analysis.group2.fit.Mratio);
    analysis.group2.Mratio.groupMean = exp(analysis.group2.fit.mu_logMratio);
    analysis.group2.Mratio.singleSubject = analysis.group2.fit.Mratio;
    analysis.groupDiff.log_Mratio.sampleDiff = analysis.group1.fit.mcmc.samples.mu_logMratio - analysis.group2.fit.mcmc.samples.mu_logMratio;
    analysis.groupDiff.log_Mratio.mean = analysis.group1.fit.mu_logMratio - analysis.group2.fit.mu_logMratio;
    analysis.groupDiff.log_Mratio.hdi = calc_HDI(analysis.groupDiff.log_Mratio.sampleDiff(:));
    analysis.groupDiff.Mratio.mean = exp(analysis.group1.fit.mu_logMratio) - exp(analysis.group2.fit.mu_logMratio);
    analysis.groupDiff.Mratio.hdi = calc_HDI((exp(analysis.group1.fit.mcmc.samples.mu_logMratio(:)) - exp(analysis.group2.fit.mcmc.samples.mu_logMratio(:))));

    % Pull out group mean values for all non-hierarchical metrics
    analysis.group1.d1.groupMean = mean(analysis.group1.fit.d1);
    analysis.group1.d1.singleSubject = analysis.group1.fit.d1;
    analysis.group1.c1.groupMean = mean(analysis.group1.fit.c1);
    analysis.group1.c1.singleSubject = analysis.group1.fit.c1;
    analysis.group2.d1.groupMean = mean(analysis.group2.fit.d1);
    analysis.group2.d1.singleSubject = analysis.group2.fit.d1;
    analysis.group2.c1.groupMean = mean(analysis.group2.fit.c1);
    analysis.group2.c1.singleSubject = analysis.group2.fit.c1;
    
    % Calculate average confidence, add filter number and accuracy to results for group 1
    for n = 1:length(analysis.group1.data)
        analysis.group1.avgConfidence.singleSubject(n) = mean(analysis.group1.data(n).results.thresholdTrials.confidence);
        analysis.group1.filterNum.singleSubject(n) = mean(analysis.group1.data(n).results.thresholdTrials.filterNum);
        analysis.group1.accuracy.singleSubject(n) = analysis.group1.data(n).results.thresholdTrials.accuracyTotal;
    end
    analysis.group1.avgConfidence.groupMean = mean(analysis.group1.avgConfidence.singleSubject);
    analysis.group1.filterNum.groupMean = mean(analysis.group1.filterNum.singleSubject);
    analysis.group1.accuracy.groupMean = mean(analysis.group1.accuracy.singleSubject);
    
    % Calculate average confidence, add filter number and accuracy to results for group 2
    for n = 1:length(analysis.group2.data)
        analysis.group2.avgConfidence.singleSubject(n) = mean(analysis.group2.data(n).results.thresholdTrials.confidence);
        analysis.group2.filterNum.singleSubject(n) = mean(analysis.group2.data(n).results.thresholdTrials.filterNum);
        analysis.group2.accuracy.singleSubject(n) = analysis.group2.data(n).results.thresholdTrials.accuracyTotal;
    end
    analysis.group2.avgConfidence.groupMean = mean(analysis.group2.avgConfidence.singleSubject);
    analysis.group2.filterNum.groupMean = mean(analysis.group2.filterNum.singleSubject);
    analysis.group2.accuracy.groupMean = mean(analysis.group2.accuracy.singleSubject);
    
    % Calculate group difference values for all non-hierarchical metrics
    analysis.groupDiff.filterNum.meanDiff = analysis.group1.filterNum.groupMean - analysis.group2.filterNum.groupMean;
    analysis.groupDiff.accuracy.meanDiff = analysis.group1.accuracy.groupMean - analysis.group2.accuracy.groupMean;
    analysis.groupDiff.d1.meanDiff = analysis.group1.d1.groupMean - analysis.group2.d1.groupMean;
    analysis.groupDiff.c1.meanDiff = analysis.group1.c1.groupMean - analysis.group2.c1.groupMean;
    analysis.groupDiff.avgConfidence.meanDiff = analysis.group1.avgConfidence.groupMean - analysis.group2.avgConfidence.groupMean;

    % Run specified statistical test on all non-hierarchical metrics
    if strcmp(analysis.groupDiff.test,'wilcoxon') == 1
        analysis.groupDiff.testType = 'Wilcoxon rank sum test';
        [analysis.groupDiff.filterNum.p,analysis.groupDiff.filterNum.h,analysis.groupDiff.filterNum.stats] = ranksum(analysis.group1.filterNum.singleSubject,analysis.group2.filterNum.singleSubject);
        [analysis.groupDiff.accuracy.p,analysis.groupDiff.accuracy.h,analysis.groupDiff.accuracy.stats] = ranksum(analysis.group1.accuracy.singleSubject,analysis.group2.accuracy.singleSubject);
        [analysis.groupDiff.d1.p,analysis.groupDiff.d1.h,analysis.groupDiff.d1.stats] = ranksum(analysis.group1.d1.singleSubject,analysis.group2.d1.singleSubject);
        [analysis.groupDiff.c1.p,analysis.groupDiff.c1.h,analysis.groupDiff.c1.stats] = ranksum(analysis.group1.c1.singleSubject,analysis.group2.c1.singleSubject);
        [analysis.groupDiff.avgConfidence.p,analysis.groupDiff.avgConfidence.h,analysis.groupDiff.avgConfidence.stats] = ranksum(analysis.group1.avgConfidence.singleSubject,analysis.group2.avgConfidence.singleSubject);
    elseif strcmp(analysis.groupDiff.test,'ttest') == 1
        analysis.groupDiff.testType = 'Unpaired T-test';
        [analysis.groupDiff.filterNum.h,analysis.groupDiff.filterNum.p,analysis.groupDiff.filterNum.ci,analysis.groupDiff.filterNum.stats] = ttest2(analysis.group1.filterNum.singleSubject,analysis.group2.filterNum.singleSubject);
        [analysis.groupDiff.accuracy.h,analysis.groupDiff.accuracy.p,analysis.groupDiff.accuracy.ci,analysis.groupDiff.accuracy.stats] = ttest2(analysis.group1.accuracy.singleSubject,analysis.group2.accuracy.singleSubject);
        [analysis.groupDiff.d1.h,analysis.groupDiff.d1.p,analysis.groupDiff.d1.ci,analysis.groupDiff.d1.stats] = ttest2(analysis.group1.d1.singleSubject,analysis.group2.d1.singleSubject);
        [analysis.groupDiff.c1.h,analysis.groupDiff.c1.p,analysis.groupDiff.c1.ci,analysis.groupDiff.c1.stats] = ttest2(analysis.group1.c1.singleSubject,analysis.group2.c1.singleSubject);
        [analysis.groupDiff.avgConfidence.h,analysis.groupDiff.avgConfidence.p,analysis.groupDiff.avgConfidence.ci,analysis.groupDiff.avgConfidence.stats] = ttest2(analysis.group1.avgConfidence.singleSubject,analysis.group2.avgConfidence.singleSubject);
    end
    
    % Save results
    save(resultsFile, 'analysis'); % Save results

    % Create Figure to display results
    figure
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 0.6 0.6]);
    labels = ['Group 1'; 'Group 2'];
    % Plot Type 1 metrics on top line
    ax1 = subplot(2,4,1);
    bar(ax1, [analysis.group1.filterNum.groupMean; analysis.group2.filterNum.groupMean]);
    hold on
    errorbar(ax1, [analysis.group1.filterNum.groupMean; analysis.group2.filterNum.groupMean], [std(analysis.group1.filterNum.singleSubject), std(analysis.group2.filterNum.singleSubject)], 'o', 'marker', 'none', 'linewidth', 1, 'Color','k');
    set(gca,'XTickLabel',labels);
    set(gca,'XTickLabelRotation',90)
    if analysis.groupDiff.filterNum.h == 0
        title('FILTER NUMBER: NON-SIG DIFF')
    elseif analysis.groupDiff.filterNum.h > 0
        title('FILTER NUMBER: SIG DIFF')
    end
    hold off
    ax2 = subplot(2,4,2);
    bar(ax2, [analysis.group1.accuracy.groupMean; analysis.group2.accuracy.groupMean]);
    hold on
    errorbar(ax2, [analysis.group1.accuracy.groupMean; analysis.group2.accuracy.groupMean], [std(analysis.group1.accuracy.singleSubject), std(analysis.group2.accuracy.singleSubject)], 'o', 'marker', 'none', 'linewidth', 1, 'Color','k');
    set(gca,'XTickLabel',labels);
    set(gca,'XTickLabelRotation',90)
    if analysis.groupDiff.accuracy.h == 0
        title('ACCURACY: NON-SIG DIFF')
    elseif analysis.groupDiff.accuracy.h > 0
        title('ACCURACY: SIG DIFF')
    end
    ax3 = subplot(2,4,3);
    bar(ax3, [analysis.group1.d1.groupMean; analysis.group2.d1.groupMean]);
    hold on
    errorbar(ax3, [analysis.group1.d1.groupMean; analysis.group2.d1.groupMean], [std(analysis.group1.d1.singleSubject), std(analysis.group2.d1.singleSubject)], 'o', 'marker', 'none', 'linewidth', 1, 'Color','k');
    set(gca,'XTickLabel',labels);
    set(gca,'XTickLabelRotation',90)
    if analysis.groupDiff.d1.h == 0
        title('d'': NON-SIG DIFF')
    elseif analysis.groupDiff.d1.h > 0
        title('d'': SIG DIFF')
    end
    hold off
    ax4 = subplot(2,4,4);
    bar(ax4, [analysis.group1.c1.groupMean; analysis.group2.c1.groupMean]);
    hold on
    errorbar(ax4, [analysis.group1.c1.groupMean; analysis.group2.c1.groupMean], [std(analysis.group1.c1.singleSubject), std(analysis.group2.c1.singleSubject)], 'o', 'marker', 'none', 'linewidth', 1, 'Color','k');
    set(gca,'XTickLabel',labels);
    set(gca,'XTickLabelRotation',90)
    if analysis.groupDiff.c1.h == 0
        title('CRITERION: NON-SIG DIFF')
    elseif analysis.groupDiff.c1.h > 0
        title('CRITERION: SIG DIFF')
    end
    hold off
    % Plot group Mratio and the difference
    ax5 = subplot(2,4,5);
    histogram(ax5, exp(analysis.group1.fit.mcmc.samples.mu_logMratio))
    xlim([0 2])
    xlabel('MRatio');
    ylabel('Sample count');
    hold on
    histogram(ax5, exp(analysis.group2.fit.mcmc.samples.mu_logMratio))
    hold off
    lgd = legend('Group 1', 'Group 2', 'Location', 'northeast');
    lgd.FontSize = 10;
    if (analysis.groupDiff.Mratio.hdi(1) > 0 && analysis.groupDiff.Mratio.hdi(2) > 0) || (analysis.groupDiff.Mratio.hdi(1) < 0 && analysis.groupDiff.Mratio.hdi(2) < 0)
        title('MRATIO: SIG DIFF')
    else
        title('MRATIO: NON-SIG DIFF')
    end
    ax6 = subplot(2,4,6);
    histogram(ax6, analysis.group1.fit.mcmc.samples.mu_logMratio)
    xlabel('log(MRatio)');
    ylabel('Sample count');
    hold on
    histogram(ax6, analysis.group2.fit.mcmc.samples.mu_logMratio)
    hold off
    lgd = legend('Group 1', 'Group 2', 'Location', 'northwest');
    lgd.FontSize = 10;
    if (analysis.groupDiff.log_Mratio.hdi(1) > 0 && analysis.groupDiff.log_Mratio.hdi(2) > 0) || (analysis.groupDiff.log_Mratio.hdi(1) < 0 && analysis.groupDiff.log_Mratio.hdi(2) < 0)
        title('LOG(MRATIO): SIG DIFF')
    else
        title('LOG(MRATIO): NON-SIG DIFF')
    end
    ax7 = subplot(2,4,7);
    histogram(ax7, analysis.groupDiff.log_Mratio.sampleDiff)
    xlabel('log(MRatio)');
    ylabel('Sample count');
    if (analysis.groupDiff.log_Mratio.hdi(1) > 0 && analysis.groupDiff.log_Mratio.hdi(2) > 0) || (analysis.groupDiff.log_Mratio.hdi(1) < 0 && analysis.groupDiff.log_Mratio.hdi(2) < 0)
        title('LOG(MRATIO) DIFF: SIG DIFF')
    else
        title('LOG(MRATIO) DIFF: NON-SIG DIFF')
    end
    hold on
    ln2 = line([analysis.groupDiff.log_Mratio.hdi(1) analysis.groupDiff.log_Mratio.hdi(1)], [0 1800]);
    ln2.Color = 'r';
    ln2.LineWidth = 1.5;
    ln2.LineStyle = '--';
    ln2 = line([analysis.groupDiff.log_Mratio.hdi(2) analysis.groupDiff.log_Mratio.hdi(2)], [0 1800]);
    ln2.Color = 'r';
    ln2.LineWidth = 1.5;
    ln2.LineStyle = '--';
    hold off
    % Plot average confidence
    ax8 = subplot(2,4,8);
    bar(ax8, [analysis.group1.avgConfidence.groupMean; analysis.group2.avgConfidence.groupMean]);
    hold on
    errorbar(ax8, [analysis.group1.avgConfidence.groupMean; analysis.group2.avgConfidence.groupMean], [std(analysis.group1.avgConfidence.singleSubject), std(analysis.group2.avgConfidence.singleSubject)], 'o', 'marker', 'none', 'linewidth', 1, 'Color','k');
    set(gca,'XTickLabel',labels);
    set(gca,'XTickLabelRotation',90)
    if analysis.groupDiff.avgConfidence.h == 0
        title('CONFIDENCE: NON-SIG DIFF')
    elseif analysis.groupDiff.avgConfidence.h > 0
        title('CONFIDENCE: SIG DIFF')
    end
    hold off

    % Print figure
    figureFile = fullfile('analysis', ['filter_task_analysis_', analysis.type]); % Create figure file name
    print(figureFile, '-dtiff');

    % Display completion message on screen
    fprintf('\n________________________________________\n\n  COMPLETED GROUP DIFFERENCE ANALYSIS\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    fprintf('\n DIFF MRATIO = %.2f (HDI: %.2f to %.2f)\n', analysis.groupDiff.Mratio.mean, analysis.groupDiff.Mratio.hdi(1), analysis.groupDiff.Mratio.hdi(2));
    if (analysis.groupDiff.Mratio.hdi(1) > 0 && analysis.groupDiff.Mratio.hdi(2) > 0) || (analysis.groupDiff.Mratio.hdi(1) < 0 && analysis.groupDiff.Mratio.hdi(2) < 0)
        fprintf('HDI DOES NOT SPAN ZERO: MRATIO DIFF SIG.\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    else
        fprintf('  HDI SPANS ZERO: MRATIO DIFF NOT SIG.\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    end
    fprintf('\n        RESULTS CAN BE FOUND IN: \n             FDT/analysis/\n________________________________________\n');
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN SESSION DIFFERENCE ANALYSIS (PAIRED) IF SPECIFIED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(analysis.type,'paired') == 1 % If session difference analysis specified
    
    % For two-session difference analysis: Print message about groups being fit together --> t-tests possible on meta-d' metrics
    fprintf('\nNOTE: Groups are fit using one hierarchical model.\nFrequentist statistics (such as t-tests) are possible on any metrics involving meta-d''.\nFrequentist statistics are applied to all type 1 metrics and average confidence values.\n');
    
    % Specify parameters
    analysis.sessionDiff.mcmc_params = fit_meta_d_params;
    
    % Fit each group using a separate hierarchical Bayesian model
    analysis.sessionDiff.fit = fit_meta_d_mcmc_groupCorr(analysis.trials2counts.nR_S1, analysis.trials2counts.nR_S2, analysis.sessionDiff.mcmc_params);
    
    % Compute HDI of difference for log(meta-d'/d')
    analysis.session1.logMratio.sessionMean = analysis.sessionDiff.fit.mu_logMratio(1);
    analysis.session1.logMratio.singleSubject = log(analysis.sessionDiff.fit.Mratio(:,1));
    analysis.session1.Mratio.sessionMean = exp(analysis.sessionDiff.fit.mu_logMratio(1));
    analysis.session1.Mratio.singleSubject = analysis.sessionDiff.fit.Mratio(:,1);
    analysis.session2.logMratio.sessionMean = analysis.sessionDiff.fit.mu_logMratio(2);
    analysis.session2.logMratio.singleSubject = log(analysis.sessionDiff.fit.Mratio(:,2));
    analysis.session2.Mratio.sessionMean = exp(analysis.sessionDiff.fit.mu_logMratio(2));
    analysis.session2.Mratio.singleSubject = analysis.sessionDiff.fit.Mratio(:,2);
    analysis.sessionDiff.log_Mratio.sampleDiff = analysis.sessionDiff.fit.mcmc.samples.mu_logMratio(:,:,1) - analysis.sessionDiff.fit.mcmc.samples.mu_logMratio(:,:,2);
    analysis.sessionDiff.log_Mratio.mean = analysis.session1.logMratio.sessionMean - analysis.session2.logMratio.sessionMean;
    analysis.sessionDiff.log_Mratio.hdi = calc_HDI(analysis.sessionDiff.log_Mratio.sampleDiff(:));
    analysis.sessionDiff.Mratio.mean = exp(analysis.session1.logMratio.sessionMean) - exp(analysis.session2.logMratio.sessionMean);
    analysis.sessionDiff.Mratio.hdi = calc_HDI((exp(analysis.sessionDiff.fit.mcmc.samples.mu_logMratio(:,:,1)) - exp(analysis.sessionDiff.fit.mcmc.samples.mu_logMratio(:,:,2))));

    % Pull out session mean values for all non-hierarchical metrics
    analysis.session1.d1.sessionMean = mean(analysis.sessionDiff.fit.d1(:,1));
    analysis.session1.d1.singleSubject = analysis.sessionDiff.fit.d1(:,1);
    analysis.session1.c1.sessionMean = mean(analysis.sessionDiff.fit.c1(:,1));
    analysis.session1.c1.singleSubject = analysis.sessionDiff.fit.c1(:,1);
    analysis.session2.d1.sessionMean = mean(analysis.sessionDiff.fit.d1(:,2));
    analysis.session2.d1.singleSubject = analysis.sessionDiff.fit.d1(:,2);
    analysis.session2.c1.sessionMean = mean(analysis.sessionDiff.fit.c1(:,2));
    analysis.session2.c1.singleSubject = analysis.sessionDiff.fit.c1(:,2);
    
    % Calculate average confidence, add filter number and accuracy to results for session 1
    for n = 1:length(analysis.session1.data)
        analysis.session1.avgConfidence.singleSubject(n) = mean(analysis.session1.data(n).results.thresholdTrials.confidence);
        analysis.session1.filterNum.singleSubject(n) = mean(analysis.session1.data(n).results.thresholdTrials.filterNum);
        analysis.session1.accuracy.singleSubject(n) = analysis.session1.data(n).results.thresholdTrials.accuracyTotal;
    end
    analysis.session1.avgConfidence.sessionMean = mean(analysis.session1.avgConfidence.singleSubject);
    analysis.session1.filterNum.sessionMean = mean(analysis.session1.filterNum.singleSubject);
    analysis.session1.accuracy.sessionMean = mean(analysis.session1.accuracy.singleSubject);
    
    % Calculate average confidence, add filter number and accuracy to results for session 2
    for n = 1:length(analysis.session2.data)
        analysis.session2.avgConfidence.singleSubject(n) = mean(analysis.session2.data(n).results.thresholdTrials.confidence);
        analysis.session2.filterNum.singleSubject(n) = mean(analysis.session2.data(n).results.thresholdTrials.filterNum);
        analysis.session2.accuracy.singleSubject(n) = analysis.session2.data(n).results.thresholdTrials.accuracyTotal;
    end
    analysis.session2.avgConfidence.sessionMean = mean(analysis.session2.avgConfidence.singleSubject);
    analysis.session2.filterNum.sessionMean = mean(analysis.session2.filterNum.singleSubject);
    analysis.session2.accuracy.sessionMean = mean(analysis.session2.accuracy.singleSubject);
    
    % Calculate session difference values for all non-hierarchical metrics
    analysis.sessionDiff.filterNum.meanDiff = analysis.session1.filterNum.sessionMean - analysis.session2.filterNum.sessionMean;
    analysis.sessionDiff.accuracy.meanDiff = analysis.session1.accuracy.sessionMean - analysis.session2.accuracy.sessionMean;
    analysis.sessionDiff.d1.meanDiff = analysis.session1.d1.sessionMean - analysis.session2.d1.sessionMean;
    analysis.sessionDiff.c1.meanDiff = analysis.session1.c1.sessionMean - analysis.session2.c1.sessionMean;
    analysis.sessionDiff.avgConfidence.meanDiff = analysis.session1.avgConfidence.sessionMean - analysis.session2.avgConfidence.sessionMean;

    % Run specified statistical test on all non-hierarchical metrics
    if strcmp(analysis.sessionDiff.test,'wilcoxon') == 1
        analysis.sessionDiff.testType = 'Wilcoxon signed rank test';
        [analysis.sessionDiff.filterNum.p,analysis.sessionDiff.filterNum.h,analysis.sessionDiff.filterNum.stats] = signrank(analysis.session1.filterNum.singleSubject,analysis.session2.filterNum.singleSubject);
        [analysis.sessionDiff.accuracy.p,analysis.sessionDiff.accuracy.h,analysis.sessionDiff.accuracy.stats] = signrank(analysis.session1.accuracy.singleSubject,analysis.session2.accuracy.singleSubject);
        [analysis.sessionDiff.d1.p,analysis.sessionDiff.d1.h,analysis.sessionDiff.d1.stats] = signrank(analysis.session1.d1.singleSubject,analysis.session2.d1.singleSubject);
        [analysis.sessionDiff.c1.p,analysis.sessionDiff.c1.h,analysis.sessionDiff.c1.stats] = signrank(analysis.session1.c1.singleSubject,analysis.session2.c1.singleSubject);
        [analysis.sessionDiff.avgConfidence.p,analysis.sessionDiff.avgConfidence.h,analysis.sessionDiff.avgConfidence.stats] = signrank(analysis.session1.avgConfidence.singleSubject,analysis.session2.avgConfidence.singleSubject);
    elseif strcmp(analysis.sessionDiff.test,'ttest') == 1
        analysis.sessionDiff.testType = 'Paired T-test';
        [analysis.sessionDiff.filterNum.h,analysis.sessionDiff.filterNum.p,analysis.sessionDiff.filterNum.ci,analysis.sessionDiff.filterNum.stats] = ttest(analysis.session1.filterNum.singleSubject,analysis.session2.filterNum.singleSubject);
        [analysis.sessionDiff.accuracy.h,analysis.sessionDiff.accuracy.p,analysis.sessionDiff.accuracy.ci,analysis.sessionDiff.accuracy.stats] = ttest(analysis.session1.accuracy.singleSubject,analysis.session2.accuracy.singleSubject);
        [analysis.sessionDiff.d1.h,analysis.sessionDiff.d1.p,analysis.sessionDiff.d1.ci,analysis.sessionDiff.d1.stats] = ttest(analysis.session1.d1.singleSubject,analysis.session2.d1.singleSubject);
        [analysis.sessionDiff.c1.h,analysis.sessionDiff.c1.p,analysis.sessionDiff.c1.ci,analysis.sessionDiff.c1.stats] = ttest(analysis.session1.c1.singleSubject,analysis.session2.c1.singleSubject);
        [analysis.sessionDiff.avgConfidence.h,analysis.sessionDiff.avgConfidence.p,analysis.sessionDiff.avgConfidence.ci,analysis.sessionDiff.avgConfidence.stats] = ttest(analysis.session1.avgConfidence.singleSubject,analysis.session2.avgConfidence.singleSubject);
    end
    
    % Save results
    save(resultsFile, 'analysis'); % Save results

    % Create Figure to display results
    figure
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 0.6 0.6]);
    labels = ['Session 1'; 'Session 2'];
    % Plot Type 1 metrics on top line
    ax1 = subplot(2,4,1);
    bar(ax1, [analysis.session1.filterNum.sessionMean; analysis.session2.filterNum.sessionMean]);
    hold on
    errorbar(ax1, [analysis.session1.filterNum.sessionMean; analysis.session2.filterNum.sessionMean], [std(analysis.session1.filterNum.singleSubject), std(analysis.session2.filterNum.singleSubject)], 'o', 'marker', 'none', 'linewidth', 1, 'Color','k');
    set(gca,'XTickLabel',labels);
    set(gca,'XTickLabelRotation',90)
    if analysis.sessionDiff.filterNum.h == 0
        title('FILTER NUMBER: NON-SIG DIFF')
    elseif analysis.sessionDiff.filterNum.h > 0
        title('FILTER NUMBER: SIG DIFF')
    end
    hold off
    ax2 = subplot(2,4,2);
    bar(ax2, [analysis.session1.accuracy.sessionMean; analysis.session2.accuracy.sessionMean]);
    hold on
    errorbar(ax2, [analysis.session1.accuracy.sessionMean; analysis.session2.accuracy.sessionMean], [std(analysis.session1.accuracy.singleSubject), std(analysis.session2.accuracy.singleSubject)], 'o', 'marker', 'none', 'linewidth', 1, 'Color','k');
    set(gca,'XTickLabel',labels);
    set(gca,'XTickLabelRotation',90)
    if analysis.sessionDiff.accuracy.h == 0
        title('ACCURACY: NON-SIG DIFF')
    elseif analysis.sessionDiff.accuracy.h > 0
        title('ACCURACY: SIG DIFF')
    end
    ax3 = subplot(2,4,3);
    bar(ax3, [analysis.session1.d1.sessionMean; analysis.session2.d1.sessionMean]);
    hold on
    errorbar(ax3, [analysis.session1.d1.sessionMean; analysis.session2.d1.sessionMean], [std(analysis.session1.d1.singleSubject), std(analysis.session2.d1.singleSubject)], 'o', 'marker', 'none', 'linewidth', 1, 'Color','k');
    set(gca,'XTickLabel',labels);
    set(gca,'XTickLabelRotation',90)
    if analysis.sessionDiff.d1.h == 0
        title('d'': NON-SIG DIFF')
    elseif analysis.sessionDiff.d1.h > 0
        title('d'': SIG DIFF')
    end
    hold off
    ax4 = subplot(2,4,4);
    bar(ax4, [analysis.session1.c1.sessionMean; analysis.session2.c1.sessionMean]);
    hold on
    errorbar(ax4, [analysis.session1.c1.sessionMean; analysis.session2.c1.sessionMean], [std(analysis.session1.c1.singleSubject), std(analysis.session2.c1.singleSubject)], 'o', 'marker', 'none', 'linewidth', 1, 'Color','k');
    set(gca,'XTickLabel',labels);
    set(gca,'XTickLabelRotation',90)
    if analysis.sessionDiff.c1.h == 0
        title('CRITERION: NON-SIG DIFF')
    elseif analysis.sessionDiff.c1.h > 0
        title('CRITERION: SIG DIFF')
    end
    hold off
    % Plot session Mratio and the difference
    ax5 = subplot(2,4,5);
    histogram(ax5, exp(analysis.sessionDiff.fit.mcmc.samples.mu_logMratio(:,:,1)))
    xlim([0 2])
    xlabel('MRatio');
    ylabel('Sample count');
    hold on
    histogram(ax5, exp(analysis.sessionDiff.fit.mcmc.samples.mu_logMratio(:,:,2)))
    hold off
    lgd = legend('Session 1', 'Session 2', 'Location', 'northeast');
    lgd.FontSize = 10;
    if (analysis.sessionDiff.Mratio.hdi(1) > 0 && analysis.sessionDiff.Mratio.hdi(2) > 0) || (analysis.sessionDiff.Mratio.hdi(1) < 0 && analysis.sessionDiff.Mratio.hdi(2) < 0)
        title('MRATIO: SIG DIFF')
    else
        title('MRATIO: NON-SIG DIFF')
    end
    ax6 = subplot(2,4,6);
    histogram(ax6, analysis.sessionDiff.fit.mcmc.samples.mu_logMratio(:,:,1))
    xlabel('log(MRatio)');
    ylabel('Sample count');
    hold on
    histogram(ax6, analysis.sessionDiff.fit.mcmc.samples.mu_logMratio(:,:,2))
    hold off
    lgd = legend('Session 1', 'Session 2', 'Location', 'northwest');
    lgd.FontSize = 10;
    if (analysis.sessionDiff.log_Mratio.hdi(1) > 0 && analysis.sessionDiff.log_Mratio.hdi(2) > 0) || (analysis.sessionDiff.log_Mratio.hdi(1) < 0 && analysis.sessionDiff.log_Mratio.hdi(2) < 0)
        title('LOG(MRATIO): SIG DIFF')
    else
        title('LOG(MRATIO): NON-SIG DIFF')
    end
    ax7 = subplot(2,4,7);
    histogram(ax7, analysis.sessionDiff.log_Mratio.sampleDiff)
    xlabel('log(MRatio)');
    ylabel('Sample count');
    if (analysis.sessionDiff.log_Mratio.hdi(1) > 0 && analysis.sessionDiff.log_Mratio.hdi(2) > 0) || (analysis.sessionDiff.log_Mratio.hdi(1) < 0 && analysis.sessionDiff.log_Mratio.hdi(2) < 0)
        title('LOG(MRATIO) DIFF: SIG DIFF')
    else
        title('LOG(MRATIO) DIFF: NON-SIG DIFF')
    end
    hold on
    ln2 = line([analysis.sessionDiff.log_Mratio.hdi(1) analysis.sessionDiff.log_Mratio.hdi(1)], [0 1800]);
    ln2.Color = 'r';
    ln2.LineWidth = 1.5;
    ln2.LineStyle = '--';
    ln2 = line([analysis.sessionDiff.log_Mratio.hdi(2) analysis.sessionDiff.log_Mratio.hdi(2)], [0 1800]);
    ln2.Color = 'r';
    ln2.LineWidth = 1.5;
    ln2.LineStyle = '--';
    hold off
    % Plot average confidence
    ax8 = subplot(2,4,8);
    bar(ax8, [analysis.session1.avgConfidence.sessionMean; analysis.session2.avgConfidence.sessionMean]);
    hold on
    errorbar(ax8, [analysis.session1.avgConfidence.sessionMean; analysis.session2.avgConfidence.sessionMean], [std(analysis.session1.avgConfidence.singleSubject), std(analysis.session2.avgConfidence.singleSubject)], 'o', 'marker', 'none', 'linewidth', 1, 'Color','k');
    set(gca,'XTickLabel',labels);
    set(gca,'XTickLabelRotation',90)
    if analysis.sessionDiff.avgConfidence.h == 0
        title('CONFIDENCE: NON-SIG DIFF')
    elseif analysis.sessionDiff.avgConfidence.h > 0
        title('CONFIDENCE: SIG DIFF')
    end
    hold off

    % Print figure
    figureFile = fullfile('analysis', ['filter_task_analysis_', analysis.type]); % Create figure file name
    print(figureFile, '-dtiff');

    % Display completion message on screen
    fprintf('\n________________________________________\n\n  COMPLETED SESSION DIFFERENCE ANALYSIS\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    fprintf('\n DIFF MRATIO = %.2f (HDI: %.2f to %.2f)\n', analysis.sessionDiff.Mratio.mean, analysis.sessionDiff.Mratio.hdi(1), analysis.sessionDiff.Mratio.hdi(2));
    if (analysis.sessionDiff.Mratio.hdi(1) > 0 && analysis.sessionDiff.Mratio.hdi(2) > 0) || (analysis.sessionDiff.Mratio.hdi(1) < 0 && analysis.sessionDiff.Mratio.hdi(2) < 0)
        fprintf('HDI DOES NOT SPAN ZERO: MRATIO DIFF SIG.\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    else
        fprintf('  HDI SPANS ZERO: MRATIO DIFF NOT SIG.\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    end
    fprintf('\n        RESULTS CAN BE FOUND IN: \n             FDT/analysis/\n________________________________________\n');
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN GROUP REGRESSION ANALYSIS IF SPECIFIED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(analysis.type,'regress') == 1 % If group regression analysis specified
    
    % Specify parameters
    analysis.groupRegression.mcmc_params = fit_meta_d_params;
    
    % Standardise covariate
    analysis.groupRegression.cov = zscore(analysis.covariate.data);

    % Fit group data all at once
    analysis.groupRegression.fit = fit_meta_d_mcmc_regression(analysis.trials2counts.nR_S1, analysis.trials2counts.nR_S2, analysis.groupRegression.cov, analysis.groupRegression.mcmc_params);
    
    % Calculate log of Mratio for single subjects values
    for n = 1:length(analysis.groupRegression.fit.Mratio)
        analysis.groupRegression.log_Mratio.singleSubject(n) = log(analysis.groupRegression.fit.Mratio(n));
    end
    
    % Pull out group and single subject values
    analysis.groupRegression.d1.groupMean = mean(analysis.groupRegression.fit.d1);
    analysis.groupRegression.d1.singleSubject = analysis.groupRegression.fit.d1;
    analysis.groupRegression.c1.groupMean = mean(analysis.groupRegression.fit.c1);
    analysis.groupRegression.c1.singleSubject = analysis.groupRegression.fit.c1;
    analysis.groupRegression.meta_d.groupMean = mean(analysis.groupRegression.fit.meta_d);
    analysis.groupRegression.meta_d.singleSubject = analysis.groupRegression.fit.meta_d;
    analysis.groupRegression.Mratio.groupMean = exp(analysis.groupRegression.fit.mu_logMratio);
    analysis.groupRegression.Mratio.singleSubject = analysis.groupRegression.fit.Mratio;
    analysis.groupRegression.Mratio.hdi = calc_HDI(exp(analysis.groupRegression.fit.mcmc.samples.mu_logMratio(:)));
    analysis.groupRegression.log_Mratio.groupMean = analysis.groupRegression.fit.mu_logMratio;
    analysis.groupRegression.log_Mratio.hdi = calc_HDI(analysis.groupRegression.fit.mcmc.samples.mu_logMratio(:));
    analysis.groupRegression.beta1.groupMean = analysis.groupRegression.fit.mu_beta1;
    analysis.groupRegression.beta1.hdi = calc_HDI(analysis.groupRegression.fit.mcmc.samples.mu_beta1(:));
    
    % Calculate average confidence, add filter number and accuracy to analysis results
    for n = 1:length(analysis.data)
        analysis.groupRegression.avgConfidence.singleSubject(n) = mean(analysis.data(n).results.thresholdTrials.confidence);
        analysis.groupRegression.filterNum.singleSubject(n) = mean(analysis.data(n).results.thresholdTrials.filterNum);
        analysis.groupRegression.accuracy.singleSubject(n) = analysis.data(n).results.thresholdTrials.accuracyTotal;
    end
    analysis.groupRegression.avgConfidence.groupMean = mean(analysis.groupRegression.avgConfidence.singleSubject);
    analysis.groupRegression.filterNum.groupMean = mean(analysis.groupRegression.filterNum.singleSubject);
    analysis.groupRegression.accuracy.groupMean = mean(analysis.groupRegression.accuracy.singleSubject);
    
    % Save results
    save(resultsFile, 'analysis'); % Save results
    
    % Display completion message on screen plus results
    fprintf('\n________________________________________\n\n  COMPLETED GROUP REGRESSION ANALYSIS\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    fprintf('\n GROUP MRATIO = %.2f (HDI: %.2f to %.2f)\n', analysis.groupRegression.Mratio.groupMean, analysis.groupRegression.Mratio.hdi(1), analysis.groupRegression.Mratio.hdi(2));
    if analysis.groupRegression.Mratio.hdi(1) > 0
        fprintf('   HDI DOES NOT SPAN ZERO: MRATIO SIG.\n');
    else
        fprintf('   HDI SPANS ZERO: MRATIO NOT SIG.\n');
    end
    fprintf('\n GROUP BETA = %.2f (HDI: %.2f to %.2f)\n', analysis.groupRegression.beta1.groupMean, analysis.groupRegression.beta1.hdi(1), analysis.groupRegression.beta1.hdi(2));
    if analysis.groupRegression.beta1.hdi(1) > 0
        fprintf('    HDI DOES NOT SPAN ZERO: BETA SIG.\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    else
        fprintf('      HDI SPANS ZERO: BETA NOT SIG.\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n');
    end
    fprintf('\n        RESULTS CAN BE FOUND IN: \n             FDT/analysis/\n________________________________________\n');
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
