function [verbose] = tapas_physio_plot_iterative_template_creation(hasFirstGuessPeaks,...
    t, c, cpulse1stGuess, nPulses1, nPulses2, cpulse2ndGuess, meanLag1, meanLag2, verbose)
% Plot refined heartrate templates
%  (C) 2023 Johanna Bayer

stringTitle = 'Preproc: Iterative Template Creation Single Cycle';

if hasFirstGuessPeaks

        fh = tapas_physio_get_default_fig_params();
        set(fh, 'Name', stringTitle);
        verbose.fig_handles(end+1) = fh;
        subplot(3,1,1);
        hold off
        hp(3) = plot(t, c, 'k');
        hold all;
        hp(1) = stem(t(cpulse1stGuess), ...
            4*ones(nPulses1,1),'b');

        hp(2) = stem(t(cpulse2ndGuess),...
            4*ones(nPulses2,1),'r');

        stringLegend = {
            sprintf('1st guess peaks (N =%d)', nPulses1), ...
            sprintf('2nd guess peaks (N =%d)', nPulses2), ...
            'raw time series'
            };

        legend(hp, stringLegend);
        title('Finding first peak (cycle start), backwards')



        %% Plot difference between detected events
        subplot(3,1,2);


        plot(t(cpulse1stGuess(2:end)), diff(t(cpulse1stGuess)));
        hold all
        plot(t(cpulse2ndGuess(2:end)), diff(t(cpulse2ndGuess)));
        title('Temporal lag between events')

         stringLegend = {
            sprintf('1st Guess (Mean lag duration %3.1f s)', meanLag1), ...
            sprintf('2nd Guess (Mean lag duration %3.1f s)', meanLag2) ...
            };

        legend(stringLegend);

else
        fh = tapas_physio_get_default_fig_params();
        verbose.fig_handles(end+1) = fh;
        subplot(3,1,1);
        plot(t, c, 'k'); title('Preproc: Finding first peak of cycle, backwards')

end
end