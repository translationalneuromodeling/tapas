function fh = tapas_physio_plot_traces(pulset, rsampint, rout, resp_max, ...
    cumsumh, sumh, h, npulse, dpulse, rphase)

nsamples = length(pulset);
t = (0:nsamples-1)*rsampint;
feqht = cumsumh/sumh*pi;

fh = tapas_physio_get_default_fig_params();
set(fh, 'Name', ...
   'Preproc: get_respiratory_phase: histogram for respiratory phase estimation');

hs(1) = subplot(2,2,1);
plot(t,pulset);
xlabel('t (s)');
ylabel('breathing amplitude (a. u.)');
title('(filtered) breathing time series');

if resp_max < inf
    hold on;
    plot(t, ones(size(t)) * resp_max, 'k--');
    hold on;
    hp = plot(t, -ones(size(t)) * resp_max, 'k--');
    legend(hp, ...
        'threshold for maximum amplitude to be considered in histogram');
    set(gcf, 'Name', ...
        [get(gcf, 'Name') ' - with amplitude overshoot-correction']);
end

hs(2) = subplot(2,2,2);
bar(rout, h);
xlabel('normalized breathing amplitude');
ylabel('counts');
title('histogram for phase mapping');
xlim([-0.1 1.1]);

hs(3) = subplot(2,2,3); plot(rout, [feqht, cos(feqht), sin(feqht)]);
xlabel('normalized breathing amplitude');
title(...
    'equalized histogram bin amplitude -> phase transfer function (f_{eqht})');
legend('f: normalized amplitude -> phase transfer function', 'cos(f)', ...
    'sin(f)', 'Location', 'NorthWest');

%figure('Name', 'Histogram: Respiration phase estimation');
hs(4) = subplot(2,2,4);
plot(t, [npulse*10, dpulse, (rphase-pi)]);
legend('10*normalized breathing belt amplitude', ...
    '-1 = exhale, 1 = inhale', 'estimated respiratory phase');
ylim([-0.2 10.2]);
title('Histogram-based respiration phase estimation');

linkaxes(hs([1 4]), 'x');
linkaxes(hs([2 3]), 'x');

end