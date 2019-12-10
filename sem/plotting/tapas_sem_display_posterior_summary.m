function tapas_sem_display_posterior_summary(results, summary)
%% Display the posterior summary. 
%
% Input
%       results     -- Results structure
%       summary     -- Summary structure of the posterior.
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

CONG = 0;
INCONG = 1;

data = results.data;

% Offset of the data
if ~isfield(data(1).y, 'offset')
    offset = 0;
else
    offset = data(1).y.offset;
end

% Assume data in milliseconds and a scaling of 100ms
if ~isfield(data(1).y, 'scale')
    scale = 100;
else
    scale = data(1).y.scale;
end

% Collapse all the data
t = arrayfun(@(x) x.y.t, data, 'UniformOutput', false);
t = vertcat(t{:});

t = scale * t + offset;

% (a)ctions
a = arrayfun(@(x) x.y.a, data, 'UniformOutput', false);
a = vertcat(a{:});

% (t)rial (t)ype
tt = arrayfun(@(x) x.u.tt, data, 'UniformOutput', false);
tt = vertcat(tt{:});

% Plot only correct trials
ttype = mod(tt, 2);
correct = a == ttype;

%% Normalized fits

fits = summary.fits;
nf = numel(fits);

% Normalized fits
nfits = cell(nf, 1);

for i = 1:numel(fits)
    nfits{i} = tapas_sem_normalized_fits(data(i), fits{i});
end

%% Display a table with the summaries
tapas_display_table(summary.summaries, 'Posterior summary')

%% Plot all the responses combined
fig = figure('name', 'Group responses and fits');
fig.Color = 'w';

all_y = struct('t', t, 'a', a);
all_u = struct('tt', tt);
edges = tapas_sem_plot_responses(all_y, all_u);
dt = edges(2) - edges(1);

plot_predicted_normalized_fits(nfits, dt, offset, scale);

%% Delta plots
fig = figure('name', 'Group delta plots');
fig.Color = 'w';

hold on
% Plot all together
tapas_sem_empirical_delta_plot(...
    t(correct & (ttype == CONG)), ...
    t(correct & (ttype == INCONG)));

xlabel('time')
ylabel('Delta incong. RT - cong. RT')
% Make delta plots
time = scale * fits{1}(1).t + offset;

nt = numel(time);

cong = zeros(nt, 1);
incong = zeros(nt, 1);

for i = 1:numel(nfits)
    for j = 1:numel(nfits{i})
        cong = cong + nfits{i}(j).pro;
        incong = incong + nfits{i}(j).anti;
    end    
end

time = reshape(time, numel(time), 1);

tapas_sem_predicted_delta_plot(time, cong, incong);
legend({'Empirical delta', 'Predicted delta'})
end

function plot_predicted_normalized_fits(nfits, dt, offset, scale)

fits = nfits{1};

for i = 2:numel(nfits)
    for j = 1:numel(fits)
        fits(j).pro = fits(j).pro + nfits{i}(j).pro;
        fits(j).anti = fits(j).anti + nfits{i}(j).anti;
    end
end

for j = 1:numel(fits)
    fits(j).t = fits(j).t * scale + offset;
end

tapas_sem_plot_fits(fits, dt/scale)

end
