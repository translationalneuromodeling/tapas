function [verbose] = tapas_physio_plot_sync_bundles(Nallvols, t, SLICELOCS, verbose)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    stringTitle =  'Sync: Slice bundles belonging to 1 volume';
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', stringTitle);
    for v=1:Nallvols-1, stem(t(SLICELOCS{v}),ones(size(SLICELOCS{v})));hold all;end
    title(stringTitle);
    xlabel('t (seconds since SCANPHYSLOG-start)');
end