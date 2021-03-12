% output = demo_image_math_imcalc_fslmaths()
%
% IN
%
% OUT
%
% EXAMPLE
%   demo_math_with_two_images
%
%   See also

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-11-18
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

clear;
close all;
clc;

doSaveForManuscript = 0;
savePath = fullfile(pwd, 'outputFigures');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load example files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathExamples = tapas_uniqc_get_path('examples');
fileTest = fullfile(pathExamples, 'nifti', 'rest', 'fmri_short.nii');

% load data
I = MrImage(fileTest);

% plot first volume
I.plot();
% plot slice 15 over time
I.plot('z', 15, 'sliceDimension', 't')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Perform Test Operations on statistical image functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

meanI       = I.mean();
meanI.name  = 'meanS';

stdI        = I.std();
stdI.name   = 'stdS';

% old compute SNR
snrI1       = I.snr();
snrI1.name  = 'snrI1';

% compute SNR via binary operation
snrI2       = meanI./stdI;
snrI2.name  = 'snrI2';

% compute SNR by hand - all meta data is lost :/
snrI3       = MrImage(mean(I.data, 4) ./ std(I.data, 0, 4));

% a somehow self-referring test, since we use perform_binary_operation :-)
deltaSnr1    = snrI2 - snrI1;
deltaSnr1.name = 'deltaSnr 1';

deltaSnr2    = snrI3 - snrI1;
deltaSnr2.name = 'deltaSnr 2';

relDeltaSnr1         = (snrI2 - snrI1)./snrI1;
relDeltaSnr1.name    = 'relDeltaSnr 1';

relDeltaSnr2         = (snrI3 - snrI1)./snrI1;
relDeltaSnr2.name    = 'relDeltaSnr 2';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Report and compare to expected results by plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fh = snrI1.plot;
if doSaveForManuscript
    fh.PaperSize = [25 35];
    saveas(fh, fullfile(savePath, 'maths_snr.pdf'));
end
snrI2.plot
snrI3.plot;

% should be all zero
deltaSnr1.plot('colorBar', 'on');
relDeltaSnr1.plot('colorBar', 'on');
deltaSnr2.plot('colorBar', 'on');
relDeltaSnr2.plot('colorBar', 'on');
% compare geometries
disp(snrI1.geometry);
disp(snrI3.geometry);
% compare info
disp('info SNR I1: '); disp(snrI1.info);
disp('info SNR I3: '); disp(snrI3.info);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do some funny image math
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% scale mean image to [0 1]
meanIScaled = (meanI - meanI.min)./meanI.max;
meanIScaled.name = 'scaled mean';
meanI.plot('colorBar', 'on');
meanIScaled.plot('colorBar', 'on');

% BTW, there is also a function (I.scale) that does this directly

% add Gaussian noise to the image time series
% make noise image with same dimensions as I
noise = MrImage(randn(I.geometry.nVoxels));
noise.name = 'random noise';
fh = noise.plot;
if doSaveForManuscript
    fh.PaperSize = [25 35];
    saveas(fh, fullfile(savePath, 'maths_noise.pdf'));
end
noiseI = I.scale + noise .* 0.05; % add just a bit noise, though
noiseI.name = 'noisy image time series';
fh = noiseI.plot;
if doSaveForManuscript
    fh.PaperSize = [25 35];
    saveas(fh, fullfile(savePath, 'maths_noisy_image.pdf'));
end

% filter using smooth
smoothedNoiseI = noiseI.smooth('fwhm', 2);
smoothedNoiseI.name = 'smoothed time series';
fh = smoothedNoiseI.plot;
if doSaveForManuscript
    fh.PaperSize = [25 35];
    saveas(fh, fullfile(savePath, 'maths_smoothed_image.pdf'));
end
% compute snr
smoothedNoiseSnr = smoothedNoiseI.snr;
fh = smoothedNoiseSnr.plot('displayRange', [0 35]);
if doSaveForManuscript
    fh.PaperSize = [25 35];
    saveas(fh, fullfile(savePath, 'maths_smoothed_image_snr.pdf'));
end

% filter using matlab 3D median filter - function handle allows any
% operation to be integrated
IMedianFilter = noiseI.perform_unary_operation(@(x) medfilt3(x), '3d');
IMedianFilter.name = 'median filtered time series';
fh = IMedianFilter.plot;
if doSaveForManuscript
    fh.PaperSize = [25 35];
    saveas(fh, fullfile(savePath, 'maths_median_filtered_image.pdf'));
end
IMedianFilter.plot('z', 15, 'sliceDimension', 't');
IMedianFilterSnr = IMedianFilter.snr;
fh = IMedianFilterSnr.plot('displayRange', [0 35]);
if doSaveForManuscript
    fh.PaperSize = [25 35];
    saveas(fh, fullfile(savePath, 'maths_median_filtered_image_snr.pdf'));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Difference of time series and Fourier analysis in space and time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fftISpace = fft(I, '2D');
fftISpace.name = 'fft of image per slice';
fftISpace.plot;

backTransformedI = ifft(fftISpace, '2D');
backTransformedI.name = 'ifft(fft) of image per slice';
backTransformedI.plot;

% perform FFT along time dimension, extract region data and plot it
fftTime = fft(I, 4);
fftTime.name = 'fft along time dimension';
fftTime.plot;
fftTime.plot('z', 15, 'sliceDimension', 't');

maskMean = meanI.compute_mask('threshold', 0.5);
maskMean.name = 'mask from mean';
maskMean.plot();
meanI.plot();

% do ROI analysis
fftTimeAbs = abs(fftTime);
fftTimeAbs.extract_rois(maskMean);
fftTimeAbs.compute_roi_stats();
fftTimeAbs.plot_rois('plotType', 'timeseries');

% now do the same on mean corrected data
IC = I - meanI;
IC.name = 'mean corrected data';
IC.plot;
IC.plot('z', 15, 'sliceDimension', 't');

fftTimeIC = fft(IC, 4);
fftTimeIC.name = 'fft of mean corrected data over time';
fftTimeIC.plot('z', 15, 'sliceDimension', 't');

% compute abs for ROI analysis
fftTimeICAbs = abs(fftTimeIC);
fftTimeICAbs.extract_rois(maskMean);
fftTimeICAbs.compute_roi_stats();
fftTimeICAbs.plot_rois('plotType', 'timeseries');