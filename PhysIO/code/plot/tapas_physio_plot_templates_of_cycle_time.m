function [verbose] = tapas_physio_plot_templates_of_cycle_time(tTemplate, ...
    template, pulseTemplate, pulseCleanedTemplate, verbose)
% (c) 2023 Johanna Bayer

    % First plot
    fh = verbose.fig_handles(end);
    figure(fh);
    subplot(3,1,3);

    plot(tTemplate, template');
    hold all;
    hp(1) = plot(tTemplate, pulseTemplate', '.--g', 'LineWidth', 3, 'Marker', ...
        'o');
    xlabel('t (seconds)');
    title('Templates of cycle time course and mean template');

    % Second plot
    stringTitle = 'Preproc: Iterative Template Creation Single Cycle';
    hp(2) = plot(tTemplate, pulseCleanedTemplate, '.-g', 'LineWidth', 4, ...
        'Marker', 'x');
    legend(hp, 'mean of templates', 'mean of most similar, chosen templates');
    tapas_physio_suptitle(stringTitle);


end