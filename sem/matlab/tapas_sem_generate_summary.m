function [summary] = tapas_sem_generate_summary(data, samples, model, subject)
%% Generate a summary table from the model.
%
% Input
%       data        -- Data from subjects.
%       samples     -- Posterior samples.
%       model       -- Model stucture.
%       subject     -- Subject identifier. Defaults to 1.
% Output
%       summary     -- Table with the summary

% aponteeduardo@gmail.com
% copyright (C) 2019
%

if nargin < 4
    subject = 1;
end

[mtype, param] = tapas_sem_identify_model(model.llh);
[funcs] = tapas_sem_get_model_functions(mtype, param);

% If necessary downsample
nsamples = size(samples, 2);

if nsamples > 30
    spacing = ceil(nsamples/30);
    samples = samples(1:spacing:end);
end

results = funcs.summaries(samples);
results = horzcat(results{:});

fields = fieldnames(results);
nconds = size(results, 1);

array = cell(nconds, numel(fields));


for i = 1:nconds
    for j = 1:numel(fields)
        array{i, j} = mean([results(i, :).(fields{j})]);
    end
end

results = struct( ...
    'subject', repmat(subject, nconds, 1), ...
    'conditions', (1:nconds)');

for i = 1:numel(fields)
    results.(fields{i}) = [array{:, i}]';
end

summary = struct2table(results);

end
