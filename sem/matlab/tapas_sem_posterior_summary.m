function [summary] = tapas_sem_posterior_summary(posterior, time)
%% Generate a summary of the posterior estimates.
%
% Input
%       posterior       -- Estimate from the model.
% Output
%       summary         -- Summary structue computed from the posterior.
%       

% aponteeduardo@gmail.com
% copyright (C) 2019
%

n = 1;

n = n + 1;
if nargin < n
    time = linspace(0, 10.0, 1200);
end 

% Plot the data.
data = posterior.data;
try
    model = posterior.model.graph{1}.htheta.model;
catch err
    try
        model = posterior.ptheta;
    catch
        error('tapas:sem:display_posterior', 'Improper input')
    end
end

ns = numel(data);

summary = struct('summaries', [], 'fits', []);

summaries = cell(ns, 1);
fits = cell(ns, 1);

for i = 1:ns
    samples = posterior.ps_theta(i, :);
    fits{i} = tapas_sem_generate_fits(data(i), samples, model, time);
    summaries{i} = tapas_sem_generate_summary(data(i), samples, model, i);
end

summary.fits = fits;
summary.summaries = vertcat(summaries{:});

end
