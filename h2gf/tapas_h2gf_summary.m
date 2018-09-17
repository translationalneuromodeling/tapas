function [summary] = tapas_h2gf_summary(samples, hgf)
%% Provides a summary of the results for the user.
%
% Input
%   pt       -- A structure obtained from tapas_h2gf_estimate
%
% Output
%   summary         -- A structure containing some relevant information for
%                       for the user.
%
% The summary is a structure array with N x 1 dimensions where N is the
% number of subjects.
%
%   prc_mean:   Mean of the perceptual parameters in the sampling space.
%   obs_mean:   Mean of the observation parameters in the sampling space.
%   mean:       Mean of the active parameters.
%   covariance: Estimated covariance of the active parameters.
%   r_hat   :   R hat statistic of the parameters
%

% aponteeduardo@gmail.com
% copyright (C) 2018
%


[nsubjects] = size(samples, 1);

subjects = struct('prc_mean', cell(nsubjects, 1), 'obs_mean', [], ...
    'mean', [], 'covariance', []);

for i = 1:nsubjects
    values = [samples{i, :}];    
    expected = mean(values, 2);
    [prc, obs] = tapas_hgf_get_theta(expected, hgf);

    subjects(i).prc_mean = prc;
    subjects(i).obs_mean = obs;
    subjects(i).covariance = cov(values');
    r_hat = psrf(values')';
    subjects(i).r_hat = r_hat;
end

summary = subjects;

end
