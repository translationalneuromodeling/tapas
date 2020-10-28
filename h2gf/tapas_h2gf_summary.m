function [summary] = tapas_h2gf_summary(data, posterior, hgf)
%% Provides a summary of the results for the user.
%
% Input
%   data        -- A structure with the data of each subject.
%   posterior   -- Posterior structure.
%   hgf         -- HGF model.
%
% Output
%   summary         -- A structure containing some relevant information for
%                       for the user.
%
% The summary is a structure array with N x 1 dimensions where N is the
% number of subjects.
%
%   y:      Observed responses
%   u:      Input to agent
%   ign:    Indices of ignored trials
%   irr:    Indices of irregular trials
%   c_prc:  Configuration of the perceptual model
%   c_prc:  Configuration of the observation model
%   c_opt:  Information on the optimization algorithm
%   optim:  Information on the optimization results
%   p_prc:  Median of the perceptual parameters in the sampling space.
%   p_obs:  Median of the observation parameters in the sampling space.
%   traj:   Trajectories of the environmental states tracked by the perceptual model
%

% aponteeduardo@gmail.com, chmathys@ethz.ch
% copyright (C) 2018
%

samples = posterior.samples.subjects;

T = posterior.T;
llh = posterior.llh;

[nsubjects] = size(samples, 1);

% Initialize structure holding an estimation summary for each subject
subjects = struct('y', cell(nsubjects, 1),...
                  'u', [],...
                  'ign', [],...
                  'irr', [],...
                  'c_prc', [],...
                  'c_obs', [],...
                  'c_opt', [],...
                  'optim', [],...
                  'p_prc', [],...
                  'p_obs', [],...
                  'traj', []);

% Construct the summary such that it can be passed as an argument to the 
% plotting and diagnostics functions of the HGF Toolbox
for i = 1:nsubjects
    % Gather data and configuration settings
    subjects(i).y = data(i).y;
    subjects(i).u = data(i).u;
    subjects(i).ign = data(i).ign;
    subjects(i).irr = data(i).irr;
    subjects(i).c_prc = hgf.c_prc;
    subjects(i).c_obs = hgf.c_obs;
    subjects(i).c_opt.algorithm = 'h2gf MCMC sampling';

    % Get the medians of parameter value samples (in estimations space)
    values = [samples{i, :}];
    expected = median(values, 2);
    [ptrans_prc, ptrans_obs] = tapas_hgf_get_theta(expected, hgf);
    
    % Transform median values to native space
    [~, subjects(i).p_prc]   = hgf.c_prc.transp_prc_fun(hgf, ptrans_prc);
    [~, subjects(i).p_obs]   = hgf.c_obs.transp_obs_fun(hgf, ptrans_obs);
    subjects(i).p_prc.p   = hgf.c_prc.transp_prc_fun(hgf, ptrans_prc);
    subjects(i).p_obs.p   = hgf.c_obs.transp_obs_fun(hgf, ptrans_obs);
    subjects(i).p_prc.ptrans = ptrans_prc';
    subjects(i).p_obs.ptrans = ptrans_obs';

    % Calculate posterior covariance and correlation
    valid = logical(sum(hgf.jm, 2));
    covariance = cov(values');
    Sigma = covariance(valid, valid);
    Sigma = tapas_nearest_psd(Sigma);
    subjects(i).optim.Sigma = Sigma;
    subjects(i).optim.Corr = tapas_Cov2Corr(Sigma);

    % Store trajectories, response predictions, and residuals for median
    % parameter values
    [subjects(i).traj, infStates] =...
        hgf.c_prc.prc_fun(subjects(i), subjects(i).p_prc.ptrans, 'trans');
 
    [~, subjects(i).optim.yhat, subjects(i).optim.res] =...
        hgf.c_obs.obs_fun(subjects(i), infStates, subjects(i).p_obs.ptrans);

    % Calculate autocorrelation of residuals
    res = subjects(i).optim.res;
    res(isnan(res)) = 0; % Set residuals of irregular trials to zero
    subjects(i).optim.resAC = tapas_autocorr(res);
    
    % Calculate the (pseudo) log-model evidence
    subjects(i).optim.LME = trapz(T(i, :), mean(llh(i, :, :), 3));
    
    % Get R-hat
    r_hat = tapas_huge_psrf(values',2);
    subjects(i).optim.r_hat = r_hat;
end

summary = subjects;

end
