function [summary] = tapas_sem_generate_summary(data, samples, model, ...
    subject, robust)
%% Generate a summary table from the model.
%
% Input
%       data        -- Data from subjects.
%       samples     -- Posterior samples.
%       model       -- Model stucture.
%       subject     -- Subject identifier. Defaults to 1.
%       robust      -- If true ignore nan and inf values. Default to true
% Output
%       summary     -- Table with the summary

% aponteeduardo@gmail.com
% copyright (C) 2019
%

n = 3;

n = n + 1;
if nargin < n
    subject = 1;
end

n = n + 1;
if nargin < n
    robust = true;
end

% Define the function to compute the mean
if robust
    fmean = @robust_mean;
else
    fmean = @mean;
end

[mtype, param] = tapas_sem_identify_model(model.llh);
[funcs] = tapas_sem_get_model_functions(mtype, param);

results = funcs.summaries(samples);
results = horzcat(results{:});

fields = fieldnames(results);
nconds = size(results, 1);

array = cell(nconds, numel(fields));


for i = 1:nconds
    for j = 1:numel(fields)
        array{i, j} = fmean([results(i, :).(fields{j})]);
    end
end

results = struct( ...
    'subject', repmat(subject, nconds, 1), ...
    'conditions', (0:nconds - 1)');

for i = 1:numel(fields)
    results.(fields{i}) = [array{:, i}]';
end

summary = struct2table(results);

end

function [m] = robust_mean(values)
% Ignore infs and nans

index = (isinf(values) | isnan(values));

% Three is no valid value
if all(index)
    m = nan;
else
    m = mean(values(~index));
end

end
