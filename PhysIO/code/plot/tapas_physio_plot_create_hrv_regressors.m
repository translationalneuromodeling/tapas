function [verbose] = tapas_physio_plot_create_hrv_regressors(sample_points, hrOut,...
    hr, t, crf, convHRV, delays,samplePointsOut, convHRVOut, verbose)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', 'Model: Regressors Heart Rate: HRV X CRF');

    subplot(2,2,1)
    plot(sample_points,hr,'r');xlabel('time (seconds)');
    title('Heart Rate');
    ylabel('beats per min (bpm)');

    subplot(2,2,2)
    plot(t, crf,'r');xlabel('time (seconds)');
    title('Cardiac response function');

    subplot(2,2,3)
    plot(sample_points, convHRV,'r');xlabel('time (seconds)');
    title('Heart rate X cardiac response function');

    subplot(2,2,4)
    [tmp, iShiftMin] = min(abs(delays));

    hp{1} = plot(samplePointsOut, hrOut,'k--'); hold all;
    hp{2} = plot(samplePointsOut, squeeze(convHRVOut(:,iShiftMin,:)),'r');
    xlabel('time (seconds)');ylabel('regessor');
    legend([hp{1}(1), hp{2}(1)], 'heart rate (bpm)', 'cardiac response regressor');
end