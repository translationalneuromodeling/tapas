% Script demo_roi_analysis
% Shows simple roi creation and analysis via MrImage-methods
%
%  demo_roi_analysis
%
%
%   See also demo_fmri_qa and demo_snr_analysis_mr_series 
%   for more detailed examples using SPM/time series functionality of
%   toolbox

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-11-18
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

%
clear;
close all;
clc;

doPlot          = true;
drawManualMask  = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
pathExamples        = tapas_uniqc_get_path('data');
pathData            = fullfile(pathExamples, 'nifti', 'rest');

fileImage           = fullfile(pathData, 'fmri_short.nii');

X = MrImage(fileImage);


% Visualize data
if doPlot
    X.plot();
    X.plot3d();
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create print, and plot some statistics of image
%  default: application along last non-zero image dim
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meanX = mean(X);
stdX = std(X);
snrX = snr(X);

if doPlot
    meanX.plot();
    stdX.plot();
    snrX.plot();
end

fprintf('Min Val of Time series \t\t\t %f \n', min(X));
fprintf('Mean Val of Time series \t\t\t %f \n', meanval(X));
fprintf('Median Val of Time series \t\t %f \n', median(X));
fprintf('Max Val of Time series \t\t %f \n', max(X));
fprintf('Max Val of Time series (slice 5-10) \t %f \n', ...
    max(X, 'z', 5:10));
fprintf('Percentile (75) of Time series \t \t %f \n', ...
    prctile(X,75));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create mask from image via threshold from statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mask90 = meanX.copyobj.compute_mask('threshold', prctile(X,90));

if doPlot
    mask90.plot();
    X.plot_overlays(mask90);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create manual mask via roi drawing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if drawManualMask
    maskManual = X.draw_mask('z', 20:2:30);
    if doPlot
      X.plot_overlays(maskManual);
    end
    masks = {mask90, maskManual};
else
    masks = mask90;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extract ROI and compute stats for SNR image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4D example
X.extract_rois(masks); % fills X.data{iSlice}
% TODO plotting does not work w/o statistics
X.compute_roi_stats(); % mean/sd/snr of all slices and whole volume

fprintf('\nROI stats per Volume \n');

nVolumes = X.geometry.nVoxels(4);
fprintf('volume \t mean \t min \t max\n')

for iVol = 1:nVolumes
    fprintf('%02d %6.1f \t %6.1f \t %6.1f \n', iVol, ...
        X.rois{1}.perVolume.mean(iVol), ...
        X.rois{1}.perVolume.min(iVol), X.rois{1}.perVolume.max(iVol))
end

if drawManualMask
    for iVol = 1:nVolumes
        fprintf('%02d %6.1f \t %6.1f \t %6.1f \n', iVol, ...
            X.rois{2}.perVolume.mean(iVol), ...
            X.rois{2}.perVolume.min(iVol), X.rois{1}.perVolume.max(iVol))
    end
end

% 3D example
snrX.extract_rois(mask90);
snrX.compute_roi_stats();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot ROI stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if doPlot
   % See also MrRoi.plot for all options
   X.rois{1}.plot('plotType', 'timeSeries');  % default for 4D
   snrX.rois{1}.plot('plotType', 'histogram', 'selectedSlices', 5:10); % default for 3D
end