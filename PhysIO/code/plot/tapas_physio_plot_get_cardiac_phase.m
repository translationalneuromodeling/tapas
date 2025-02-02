function [verbose] = tapas_physio_plot_get_cardiac_phase(scannert,cardiac_phase, pulset, ...
    svolpulse, verbose)

    % 1. plot chosen slice start event
    % 2. plot chosen c_sample phase on top of chosen slice scan start, (as a stem
    % and line of phases)
    % 3. plot all detected cardiac r-wave peaks
    % 4. plot volume start event
    stringTitle = 'Preproc: tapas_physio_get_cardiac_phase: scanner and R-wave pulses - output phase';
    fh = tapas_physio_get_default_fig_params();
    set(fh, 'Name', stringTitle);
    stem(scannert, cardiac_phase, 'k'); hold on;
    plot(scannert, cardiac_phase, 'k');
    stem(pulset,3*ones(size(pulset)),'r', 'LineWidth',2);
    stem(svolpulse,7*ones(size(svolpulse)),'g', 'LineWidth',2);
    legend('estimated phase at slice events', ...
        '', ...
        'heart beat R-peak', ...
        'scan volume start');
    title(regexprep(stringTitle,'_', '\\_'));
    xlabel('t (seconds)');
    %stem(scannertpriorpulse,ones(size(scannertpriorpulse))*2,'g');
    %stem(scannertafterpulse,ones(size(scannertafterpulse))*2,'b');

end