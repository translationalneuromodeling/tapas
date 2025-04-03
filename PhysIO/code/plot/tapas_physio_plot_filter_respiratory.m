function [verbose, handles, labels] = tapas_physio_plot_filter_respiratory(rpulset,m, s, t, ...
    rpulset_out, rpulset_out_trend, trend, rpulset_out_trend_filt, verbose)
% plot repiratory filtering
% (C) 2023 Johanna Bayer
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', 'Preproc: Respiratory filtering');
    hold on;
    handles = []; labels = {};
    plot([t(1), t(end)], [0.0, 0.0], 'Color', [0.7, 0.7, 0.7]);
    handles(end+1) = plot(t, (rpulset - m) / s);
    labels{end+1} = 'Raw respiratory signal';

    % without outliers
    handles(end+1) = plot(t, (rpulset_out - m) / s);
    labels{end+1} = '... without outliers';

    % detrend
    figure(verbose.fig_handles(end));
    handles(end+1) = plot(t, (trend - m) / s);
    labels{end+1} = '... low frequency trend';
    plot([t(1), t(end)], [-5.0, -5.0], 'Color', [0.7, 0.7, 0.7]);
    handles(end+1) = plot(t, (rpulset_out_trend - m) / s - 5.0);
    labels{end+1} = '... detrended';

    % Low pass filtered to remove noise
    handles(end+1) = plot(t, (rpulset_out_trend_filt - m) / s - 5.0);
    labels{end+1} = '... after low-pass filter';
    xlim([t(1), t(end)]);
    xlabel('Time (s)');
    yticks([]);
    legend(handles, labels);

end