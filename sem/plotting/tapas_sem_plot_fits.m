function tapas_sem_plot_fits(data, fits, dt)
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

if nargin < 3
    dt = 1;
end

conds = unique(data.u.tt);
nconds = numel(conds);

for i = 1:nconds
    tt = conds(i) == data.u.tt;
    mass = dt * sum(tt);

    ax = subplot(nconds, 2, (i - 1) * 2 + 1);
    hold on;
    
    plot(fits(i).t, fits(i).pro * mass, 'k')

    ax = subplot(nconds, 2, (i - 1) * 2 + 2);
    hold on;
    plot(fits(i).t, fits(i).anti * mass, 'k')   
end


end
