function [verbose, t] = tapas_physio_plot_filter_respiratory(rpulset,rsampint, verbose)
% plot repiratory filtering 
% (C) 2023 Johanna Bayer
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', 'Preproc: Respiratory filtering');
    hold on;
    handles = []; labels = {};
    t = linspace(0.0, rsampint * (length(rpulset) - 1), length(rpulset));
    plot([t(1), t(end)], [0.0, 0.0], 'Color', [0.7, 0.7, 0.7]);
    m = mean(rpulset); s = std(rpulset);
    handles(end+1) = plot(t, (rpulset - m) / s);
    labels{end+1} = 'Raw respiratory signal';
end