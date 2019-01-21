function [] = tapas_sem_display_posterior(posterior)
%% Displays a summary of the posterior estimates.
%
% Input
%       posterior       -- Estimate from the model.
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2019
%

% Plot the data.
data = posterior.data;
model = posterior.model.graph{1}.htheta.model;

for i = 1:numel(data)
    figure
    [edges] = tapas_sem_plot_antisaccades(data(i).y, data(i).u);
    dt = edges(2) - edges(1);
    samples = posterior.ps_theta(i, :);
    fits = tapas_sem_generate_fits(data(i), samples, model);
    tapas_sem_plot_fits(data(i), fits, dt)
end

end
