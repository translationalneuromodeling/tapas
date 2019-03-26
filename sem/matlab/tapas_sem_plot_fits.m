function tapas_sem_plot_fits(data, fits, dt)
%% 
%
% Input
%
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
    
    plot(fits(i).t, exp(fits(i).pro) * mass, 'k')

    ax = subplot(nconds, 2, (i - 1) * 2 + 2);
    hold on;
    plot(fits(i).t, exp(fits(i).anti) * mass, 'k')   
end


end
