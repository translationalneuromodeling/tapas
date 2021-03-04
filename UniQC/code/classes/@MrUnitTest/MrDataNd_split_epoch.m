function this = MrDataNd_split_epoch(this)
%Tests split_epoch by creating artificial trials for fmri_short example
%
%   Y = MrUnitTest()
%   run(Y, 'MrDataNd_split_epoch');
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDataNd_split_epoch
%
%   See also MrUnitTest
 
% Author:   Lars Kasper
% Created:  2019-03-25
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

doPlotRoi = 0;
doPlotTrialsActiveVoxel = true; % plot active voxel for all events

%% Load short fMRI example
pathExample = tapas_uniqc_get_path('examples');
%pathExample = fullfile(pathExample, 'nifti', 'rest');
%fileExample = fullfile(pathExample, 'fmri_short.nii');
pathExample = fullfile(pathExample, 'nifti', 'paradigm_visual');
fileExample = fullfile(pathExample, 'fmri.nii.gz');
fileBehav = fullfile(pathExample, 'behav.mat');
x = MrImage(fileExample);

TR = 3; % s

% update erroneous TR
x.dimInfo.set_dims('t', 'resolutions', TR, 'samplingWidths', TR, ...
    'firstSamplingPoint', 0);

%% Create mask for time series evaluation; central voxel in all slices

nSamplesXY = x.dimInfo.nSamples({'x', 'y'});

M = x.select('t',1);

%iActiveVoxelXY = round([nSamplesXY/2, nSamplesXY/2]);
idxActiveVoxel = [67 109 7]; % visual voxel
M.data(:) = 0;

% spherical ROI around active voxel
M.data(idxActiveVoxel(1), idxActiveVoxel(2), idxActiveVoxel(3)) = 1;
M = M.imdilate(strel('sphere', 5), '3D');


%% Split into epochs
% alternative [onsets, durations, names] = ...
%        get_multiple_conditions_visual('bevav.log')
% and use onsets{1} and durations{1} for the 10 blocks
load(fileBehav);
onsetTimes = relativeTimeBlockStartSeconds(1:4:end); % first blocks of same kind

newPeriStimulusOnsets = 10; % number of bins, if single number, duration of stimulation block: 14.925 s

y = x.split_epoch(onsetTimes, newPeriStimulusOnsets);

%% Plot all PST time courses from active voxel
if doPlotTrialsActiveVoxel
    stringTitle = sprintf('All trials from active voxel [%d %d %d]', ...
        idxActiveVoxel(1), idxActiveVoxel(2), idxActiveVoxel(3));
    figure('Name', stringTitle);
    ally = squeeze(y.data(idxActiveVoxel(1),idxActiveVoxel(2),idxActiveVoxel(3),:,:));
    plot(y.dimInfo.samplingPoints{4}, ally);
    hold all;
    meany = mean(ally,2);
    stdy = std(ally,0,2);
    plot(y.dimInfo.samplingPoints{4}, meany, 'g-', 'LineWidth',4)
    plot(y.dimInfo.samplingPoints{4}, meany-stdy,'k:', 'LineWidth',2)
    plot(y.dimInfo.samplingPoints{4}, meany+stdy,'k:', 'LineWidth',2)
    title(stringTitle);
end

%% evaluate time series: extract roi from both raw and epoched data
x.extract_rois(M);
x.compute_roi_stats();

y.extract_rois(M);
y.compute_roi_stats();


% plot with corresponding time vector
if doPlotRoi
    x.rois{1}.plot()
    y.rois{1}.plot()
end


%% Actual unit test

expSolution = 0;
actSolution.data = 0;

absTol = 10e-7;



%% verify equality of expected and actual solution
% import matlab.unittests to apply tolerances for objects
this.verifyEqual(actSolution.data, expSolution, 'absTol', absTol);
