function this = MrDataNd_shift_timeseries(this, verboseLevel)
%Tests shift_timeseries using synthetic data (sine waves of different frequency
% per slice)
%
%   Y = MrUnitTest()
%   run(Y, 'MrDataNd_shift_timeseries')
%
% This is a method of class MrUnitTest.
%   It creates sine waves of different frequencies per slice, and does the
%   time shift with shift_timeseries and with an analytic shift in the
%   creation, respectively.
%   Also comes with detailed plotting of raw and shifted time series, plus
%   error to analytic solution, both plotting vs volume x-axis and real 
%   time vector.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDataNd_shift_timeseries
%
%   See also MrUnitTest MrUnitTest.MrDataNd_value_operation

% Author:   Lars Kasper
% Created:  2019-03-24
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% 0 = no plots, 1 = shift vs raw time series plot, 2 = indvididual MrRoi.plot
if nargin < 2
    verboseLevel = 0;
end

doPlotRoi = verboseLevel >=2;
doPlot = verboseLevel >=1;


%% Changeable parameters for sine simulation

resolutionXY    = 3; %mm
nSamplesXY      = 32;
nFrequencies    = 4; % one frequency per slice, 0:.5:(nFreq/2-.5) full periods within duration of experiment
nVolumes        = 128;
TR              = 3;
dt              = 2; % in seconds
nMarginSamples  = 8;

idxIgnoreSamples = [1:nMarginSamples nVolumes+((-nMarginSamples+1):0)];
idxTestSamples  = setdiff(1:nVolumes,idxIgnoreSamples);

%% Create raw data matrix for operation (shift_timeseries)
% array of sine frequencies
dimInfo = MrDimInfo('nSamples', [nSamplesXY nSamplesXY nFrequencies nVolumes], ...
    'resolutions', [resolutionXY, resolutionXY, 0.5, TR], ...
    'firstSamplingPoint', [resolutionXY/2, resolutionXY/2 0, 0]);

dataMatrixX = zeros(nVolumes, nFrequencies);
% to match slice-wise structure of MrRoi
t = dimInfo.t.samplingPoints{1}';
fArray = 0:0.5:(nFrequencies/2-0.5);
for iFreq = 1:nFrequencies
    dataMatrixX(:,iFreq) = sin(t/(TR*nVolumes)*2*pi*(fArray(iFreq)));
    % left-shift sine explicitly via time shift; row vector for
    % MrRoi compatibility
    expSolution{iFreq} = sin((t.'-dt)/(TR*nVolumes)*2*pi*(fArray(iFreq)));
end

dataMatrixX = repmat(permute(dataMatrixX, [3 4 2 1]), ...
    nSamplesXY, nSamplesXY, 1, 1);

%% Create expected solution: analytically shifted sine time series
expSolution = cell(nFrequencies,1);
for iFreq = 1:nFrequencies
    % left-shift sine explicitly via time shift; row vector for
    % MrRoi compatibility
    expSolution{iFreq} = sin((t.'-dt)/(TR*nVolumes)*2*pi*(fArray(iFreq)));
end


%% Create 4D image from dataMatrix with sinusoidal modulation 
% of different frequency per slice

% should be dataNd, but ROI tests easier on MrImage
x = MrImage(dataMatrixX, 'dimInfo', dimInfo);
x.name = 'raw time series';

%% Compute actual solution: Shift time series and compare in predefined ROIs
y = x.shift_timeseries(dt);
y.name = 'shifted time series';

% define mask of one central voxel, over all slices
M = x.select('t',1);
iMaskVoxelXY = round([nSamplesXY/2, nSamplesXY/2]);
M.data(:) = 0;
M.data(iMaskVoxelXY(1), iMaskVoxelXY(2), :) = 1;

% extract roi from both raw and shifted data
x.extract_rois(M);
x.compute_roi_stats();

y.extract_rois(M);
y.compute_roi_stats();

actSolution = y.rois{1};

% plot with corresponding time vector
if doPlotRoi
    x.rois{1}.plot()
    y.rois{1}.plot()
end

%% plot actual and expected solution and difference together;
if doPlot
    stringSupTitle{1} = sprintf('shift timeseries (time axis): Joint plot before/after dt = %.2f s (TR = %.2f s)', dt, TR);
    stringSupTitle{2} = sprintf('shift timeseries (volum axis): Joint plot before/after dt = %.2f s (TR = %.2f s)', dt, TR);
    for iFig = 1:2
        fh(iFig) = figure('Name', stringSupTitle{iFig}, 'WindowStyle', 'docked');
    end
    nCols = ceil(sqrt(nFrequencies));
    nRows = ceil(nFrequencies/nCols);
    t_x = x.dimInfo.t.samplingPoints{1}';
    t_y = y.dimInfo.t.samplingPoints{1}';
    for iFig = 1:2
        for iFreq = 1:nFrequencies
            figure(fh(iFig))
            hs = subplot(nRows, nCols, iFreq);
            stringTitle = sprintf('f = %.1f cycles per run', fArray(iFreq));
            
            if iFig == 1
                plot(t_x, x.rois{1}.data{iFreq}, 'o-'); hold all;
                plot(t_y, y.rois{1}.data{iFreq}, 'x-');
                plot(t_y, expSolution{iFreq,:}.', 'd-.');
                plot(t_y, expSolution{iFreq,:}.' - y.rois{1}.data{iFreq}.', 's:');
                xlabel('t (s)');
            else
                plot(x.rois{1}.data{iFreq}, 'o-'); hold all;
                plot(y.rois{1}.data{iFreq}, 'x-');
                plot(expSolution{iFreq,:}.', 'd-.');
                plot(expSolution{iFreq,:}.' - y.rois{1}.data{iFreq}.', 's:');
                xlabel('volumes');
            end
            if iFreq == 1
                legend(hs, 'raw', sprintf('shifted by %.2f s', dt), ...
                    'analytical solution', 'delta: analytical - shifted'); end
            title(stringTitle);
        end
        
        % put super title or subplot grid title above all, if
        % functions exist
        if exist('suptitle')
            suptitle(stringSupTitle{iFig});
        elseif exist('sgtitle')
            sgtitle(stringSupTitle{iFig});
        end
    end
end

%% Verify equality on subpart of samples

% very genereous because of time interval edge effects in FFT
% usually the first value is really bad!
absTol = 1e-3;

% crop to non-margin samples that have to be correct
actSolution = cellfun(@(x) x(idxTestSamples), actSolution.data, ...
    'UniformOutput', false);
expSolution = cellfun(@(x) x(idxTestSamples), expSolution, ...
    'UniformOutput', false);

% Verify equality of expected and actual solution
% import matlab.unittests to apply tolerances for objects
this.verifyEqual(actSolution, expSolution, 'absTol', absTol);
