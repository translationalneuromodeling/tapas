% Script demo_fmri_qa
% Performs quality assurance analysis on raw fMRI time series
%
%  See also MrImage MrSeries

% Author:   Sandra Iglesias & Lars Kasper
% Created:  2015-08-13
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NOTE: This script is structured into sections
% Run one section at a time, scrutinize the output plots, and only
% afterwards execute the next section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set(0, 'DefaultFigureWindowStyle', 'docked');
% warning('off', 'images:imshow:magnificationMustBeFitForDockedFigure');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Specify fmri file and parameters
% Allowed file-types for fMRI time series are: nii, img, mat, par/rec, ...
% Hint: perform this script twice, once without and once with realignment to
%        study the impact of realignment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;


doSaveForManuscript = 0;
savePath = fullfile(pwd, 'outputFigures');

doRealign           = true;
doSaveResults       = false;
doInteractive       = true;

% # MOD: the next 3 lines are to find example data only. If you have your
% own data, just replace the file name in the 3rd line, e.g.

example             = 'short'; % 'short' or 'sandra';
pathExamples        = tapas_uniqc_get_path('examples');
switch lower(example)
    case 'short'
        pathData            = fullfile(pathExamples, 'nifti', 'rest');
        fileRaw             = fullfile(pathData, 'fmri_short.nii');
    case 'sandra'
        pathData            = fullfile(pathExamples, 'siemens_prisma_3t');
        fileRaw             = fullfile(pathData, 'tSANTASK_3660S150527_151311_0010_ep2d_bold_physio_2mm_fov224_part2.nii');
end

dirResults          = 'results';

% some specific plotting options for difference images later on
selectedSlicesCor   = 50:59;
selectedSlicesSag   = 32:46;
selectedSlicesTra   = [1 17 33];
selectedVolumes     = 8:14; % Sandra: 317:325


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load raw data in MrSeries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S = MrSeries(fileRaw);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute time series statistics (mean, sd, snr) and plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S.compute_stat_images();

fh = S.mean.plot('colorBar', 'on');
if doSaveForManuscript, saveas(fh, fullfile(savePath, 'QA_mean.pdf')); end
S.sd.plot('colorBar', 'on');
S.snr.plot('colorBar', 'on');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Realign time series, if specified
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if doRealign
    S.realign();
    dirResults = [dirResults '_realigned'];
    fh = S.glm.plot_regressors('realign');
    if doSaveForManuscript, saveas(fh, fullfile(savePath, 'QA_realign.pdf')); end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Compute time series statistics (mean, sd, snr) and plot, if realign specified
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    S.compute_stat_images();
    
    S.mean.plot('colorBar', 'on');
    S.sd.plot('colorBar', 'on');
    S.snr.plot('colorBar', 'on');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Some more detailed plots, different orientations, interactive plot in SPM
% Note: if no slice selection is specified, the montage plot includes all slices
%       The rotate90 parameter only changes the display, not the data
%       itself
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fh = S.sd.plot('sliceDimension',1, 'x', selectedSlicesSag, ...
    'rotate90', 2)
if doSaveForManuscript, saveas(fh, fullfile(savePath, 'QA_SD_sag.pdf')); end
S.sd.plot('sliceDimension',2, 'y', selectedSlicesCor, ...
    'rotate90', 1)

if doInteractive
    S.sd.plot('plotType', 'spmi') % plot interactive with SPM, press Enter, when done
else
    S.sd.plot('plotType', 'spm') % plot interactive with SPM, press Enter, when done
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find outliers: Difference Images
% Explore interactive plot by changing windowing tresholds and play movies
% over dynamics (volumes) for different slices;
% Typically, some movement, e.g. in phase encoding direction should be
% discernible over the course of an fMRI session
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if doInteractive
    S.data.plot('useSlider', true);
end

% mean difference image
% alternative plot(mean(diff(S.data)))
fh = S.data.diff.mean.plot;
if doSaveForManuscript, saveas(fh, fullfile(savePath, 'QA_diff_mean.pdf')); end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find outliers: Plot Difference Image interactively
% Also: Plot some suspicious volumes zoomed.
%       Save data to nifti file.
% Plot difference images to discern fast changes (between consecutive volumes)
% e.g. sudden head movements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

diffData = S.data.diff();
diffData.name = ['Difference Images (N+1 - N) (' S.name ')'];
displayRange = [-S.mean.prctile(75), S.mean.prctile(75)];

if doInteractive
    diffData.plot('useSlider', true);
    if doSaveForManuscript, saveas(gca, fullfile(savePath, 'QA_slider.pdf')); end
end

% plot some difference volumes in detail, with custom display range
diffData.plot('t', selectedVolumes, ...
    'displayRange',  displayRange)

diffData.parameters.save.path = dirResults; ...
    diffData.parameters.save.fileName = [tapas_uniqc_str2fn(diffData.name), '.nii'];
diffData.save;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find outliers: Plot Differences to mean interactively
% Plot difference to mean for first and last image in time series to
% discern slower changes, e.g. drifts
% Some of these changes for tranverse slice acquisition can best be viewed
% in a sagittal geometry, which is shown here as well
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

diffMean = S.data - S.mean;
diffMean.name = ['Difference to mean (' S.name ')' ];
% plot difference to mean for first and last image in time series to
% discern slow changes, e.g. drifts
diffMean.plot('t', [1, diffMean.geometry.nVoxels(end)], ...
    'displayRange', displayRange);

diffMean.plot('t', [1, diffMean.geometry.nVoxels(end)], 'displayRange', ...
    displayRange, 'sliceDimension', 2, 'y', selectedSlicesCor, ...
    'rotate90', 1)


diffMean.parameters.save.path = dirResults;
diffMean.parameters.save.fileName = [tapas_uniqc_str2fn(diffMean.name), '.nii'];
diffMean.save;

% interactive plot
if doInteractive
    diffMean.plot('useSlider', true);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Next 2 sections: ROI Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute masks from mean via relative intensity thresholding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S.parameters.compute_masks.nameInputImages = 'mean';
S.parameters.compute_masks.nameTargetGeometry = 'mean';
% mask on mean >= median pixel value
S.parameters.compute_masks.threshold = S.mean.prctile(75);
S.parameters.compute_masks.keepExistingMasks = false;

S.compute_masks();

S.masks{1}.plot();

S.masks{2} = S.masks{1}.imclose;
fh = S.masks{2}.plot;
if doSaveForManuscript, saveas(fh, fullfile(savePath, 'QA_mask.pdf')); end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extract region of interest data for masks from time series data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify extraction from data!
S.parameters.analyze_rois.nameInputImages = {'data', 'snr'};
S.parameters.analyze_rois.nameInputMasks = '.*mask';
S.parameters.analyze_rois.keepCreatedRois = false;
S.analyze_rois();

% plot mean time series signal from extracted roi for all slices, and some
% more statistics for lower, upper and middle slice

S.data.rois{1}.plot('statType', 'mean');

% plot mean+sd for specific slices only, shaded plot
fh = S.data.rois{1}.plot('statType', 'mean+sd', ...
    'selectedSlices', selectedSlicesTra);
if doSaveForManuscript, saveas(fh, fullfile(savePath, 'QA_mean_sd.pdf')); end

% plot more statistics for selected slices and the whole volume
S.data.rois{1}.plot('statType', {'min', 'median', 'max'}, ...
    'selectedSlices', selectedSlicesTra);

disp(S.snr.rois{1}.perVolume.median);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Perform spatial PCA
% Plot first 3 principal components (eigenimages) and accompanying
% projection weights (representative time series from an ROI)
% Note: All individual voxel time series of the 4D "PC4D"-image
% will be just scaled versions of the same projection time course,
% since the PC images are in fact generatied as PC4D_n = PC*projection_n
% for the n-th volume
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number of principal components to be extracted
% Specify nComponents < 1 to extract a number of components that explains
% at least value*100 % of the variance in the time series
nComponents = 3;

PC4D        = S.data.pca(nComponents);

% Create same mask as above, but in a different way to show versatility
M = S.data.mean.compute_mask('threshold', 0.95);

% Create PC4D.rois{1} to extract individual time series
nComponents = numel(PC4D); % determined new, if variance threshold for PC
for c = 1:nComponents
    PC4D{c}.analyze_rois(M);
    PC4D{c}.rois{1}.name = ...
        sprintf('Time course (projection) of spatial PC %d ', c);
end

% Plot PC and corresponding projections next to each other
for c = 1:nComponents
    fh = PC4D{c}.abs.plot();
    if doSaveForManuscript, saveas(fh, fullfile(savePath, ...
            ['QA_PC', num2str(c), '_abs.pdf'])); end
    fh = PC4D{c}.rois{1}.plot('statType', 'mean', 'dataGrouping', 'volume');
    if doSaveForManuscript, saveas(fh, fullfile(savePath, ...
            ['QA_PC', num2str(c), '_time.pdf'])); end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save images to figures folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if doSaveResults
    tapas_uniqc_save_fig('fh', 'all', 'imageType', 'fig', 'pathSave', dirResults);
    tapas_uniqc_save_fig('fh', 'all', 'imageType', 'png', 'pathSave', dirResults);
end