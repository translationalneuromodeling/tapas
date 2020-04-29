function tapas_sem_plot_fits(fits, dt)
%% Plot the fits of the model.
%
% Input
%       data        -- Structure of the data of a single subject.
%       fits        -- Fits struct array
%       dt          -- Factor to normalize the area
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

n = 1;

n = n + 1;
if nargin < n
    dt = 1;
end

nconds = numel(fits);

for i = 1:nconds

    ax = subplot(nconds, 2, (i - 1) * 2 + 1);
    hold on;

    plot(fits(i).t, fits(i).pro * dt, 'k', 'linewidth', 2)

    ax = subplot(nconds, 2, (i - 1) * 2 + 2);
    hold on;
    plot(fits(i).t, fits(i).anti * dt, 'k', 'linewidth', 2)
end


end
