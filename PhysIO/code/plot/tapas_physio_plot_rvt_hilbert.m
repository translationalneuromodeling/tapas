function [verbose] = tapas_physio_plot_rvt_hilbert(t,fr, fr_lp, fr_mag, fr_rv, ...
    fr_phase, fr_if, verbose);
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', 'Model: Hilbert RVT (Respiratory Volume per Time)');

    hs(1) = subplot(2,1,1);
    hold on;
    plot([t(1), t(end)], [0.0, 0.0], 'Color', [0.7, 0.7, 0.7]);
    hp(1) = plot(t, fr);
    hp(2) = plot(t, fr_lp);
    hp(3) = plot(t, fr_mag);
    hp(4) = plot(t, fr_rv / 2.0);
    strLegend = {
        'Filtered breathing signal', ...
        '... after low pass-filter', ...
        'Breathing signal envelope', ...
        'Respiratory volume'};
    legend(hp, strLegend);
    xlim([t(1), t(end)]);
    title('Respiratory Volume (from Hilbert Transform)');

    hs(2) = subplot(2,1,2);
    hold on
    plot([t(1), t(end)], [0.0, 0.0], 'Color', [0.7, 0.7, 0.7]);
    hp(1) = plot(t, fr);
    hp(2) = plot(t, fr_lp);
    hp(3) = plot(t, std(fr) * cos(fr_phase));
    hp(4) = plot(t, fr_if);
    strLegend = {
        'Filtered breathing signal', ...
        '... after low pass-filter', ...
        '... after removing amplitude', ...
        'Instantaneous breathing rate'};
    legend(hp, strLegend);
    xlim([t(1), t(end)]);
    title('Instantaneous Breathing Rate (from Hilbert Transform)');
    linkaxes(hs, 'x')
end