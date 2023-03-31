function [verbose] = tapas_physio_preproc_peak_detection_from_automatically_generated(t,c,...
    cpulse, verbose)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
        verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
        stringTitle = 'Preproc: Peak Detection from Automatically Generated Template';
        set(gcf, 'Name', stringTitle);
        plot(t, c, 'k');
        hold all;
        stem(cpulse,4*ones(size(cpulse)), 'r');
        legend('Raw time course',...
        'Detected maxima (cardiac pulses / max inhalations)');
        title(stringTitle);
end

