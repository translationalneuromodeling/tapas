function [verbose] = tapas_physio_plot_create_scan_timing_philips(t, y, ...
    VOLLOCS, Ndummies, LOCS, LOC_END_MARKER)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
set(gcf,'Name', 'Sync: Thresholding Gradient for slice acq start detection');
fs(1) = subplot(1,1,1);
plot(t, y(:,7:9));
legend('gradient x', 'gradient y', 'gradient z');
title('Raw Gradient Time-courses');
hold on,
ylims = ylim;

plot( [(VOLLOCS(1)-1)/TA    (VOLLOCS(1)-1)/TA]  , ylims, 'k' )
plot( [(VOLLOCS(1+Ndummies)-1)/TA    (VOLLOCS(1+Ndummies)-1)/TA]  , ylims, 'g' )
plot( [(VOLLOCS(end)-1)/TA  (VOLLOCS(end)-1)/TA], ylims, 'k' )
plot( [(LOCS(end)-1)/TA     (LOCS(end)-1)/TA]   , ylims, 'k' )
plot( [(VOLLOCS(end-1)-1)/TA     (VOLLOCS(end-1)-1)/TA]   , ylims, 'k' )

plot( [(LOC_END_MARKER-1)/TA (LOC_END_MARKER-1)/TA], ylims, 'g' )
end