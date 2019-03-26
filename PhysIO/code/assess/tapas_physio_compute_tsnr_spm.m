function [tSnrImage, fileTsnr, tSnrRatioImage, fileTsnrRatio] = ...
    tapas_physio_compute_tsnr_spm(SPM, iC, iCForRatio, doInvert, doSaveNewContrasts)
% Computes temporal SNR image after correcting for a contrast
% from SPM general linear model
%
%   tSnrImage = compute_tsnr_spm(input)
%
%   This function executes the following steps:
%   1.  Compute F-Contrasts of the kind "All-but-selected regressors"
%      	run spm_write_residuals to retrieve Y-Yc = Y0 + e
%       (note that roles of Y0 and Yc are swapped compared to in spm_spm,
%       since we are interested in Residuals that do *not* stem from our
%       regressors, but the other ones!
%   2.  Compute tSNR images using mean(Residuals)/std(Residuals) from
%       F-contrasts from (1.), save as nii-files tsnr_con<iC>.nii
%   3.  Compute tSNR for contrast of comparison, default:
%       preprocessed time-series, after pre-whitening and
%       highpass-filtering (i.e. K*W*Y), saved as nii-files
%       using a recursive call to this function
%   4.  Compute tSNR ratio images from (3.) and (2.), save as nii-files
%       tsnr_con<iC>vs<iCRatio>.nii
%
% IN
%   SPM     SPM variable (in SPM.mat, or file name) after parameter and
%           contrast estimation
%   iC      Contrast of interest for which tSNR is computed
%           iC = 0 computes tSNR of the raw image time series after
%               pre-whitening and filtering, but without any model
%               confound regressors removed. (default)
%           iC = NaN computes tSNR after removing the full model (including
%               the mean), and is therefore equivalent to sqrt(1/ResMS)
%   iCForRatio
%           For computation of relative tSNR.
%           Contrast index whose tSNR image should be used as a
%           denominator for tSnrRatioImage = tSNRiC./tSNRiCForRatio;
%           tSNR is compared to the tSNR image of the specified contrast
%           (0 for raw tSNR). The corresponding tSNR-image will be created
%           default = 0 (raw tSNR); Leave [] to not compute ratio of tSNRs
%   doInvert true (default for iC > 0 and ~nan) or false
%           typically, one is interested in the tSNR *after* correcting for
%           the contrast in question. To compute this, one has to look at
%           the residuals of the inverse F-contrast that excludes all but
%           the contrast of interest, i.e. eye(nRegressors) - xcon
%           Thus, this function computes this contrast per default and goes
%           from there determining residuals etc.
%   doSaveNewContrasts
%           true or false (default)
%           if true, the temporary contrasts created by inversion of
%           selected columns will be saved into the SPM structure.
%           Otherwise, all temporary data will be removed.
%
% OUT
%   tSnrImage   [nX,nY,nZ] image matrix holding tSNR when only including regressors in
%               contrast iC in design matrix;
%               i.e. gives the effect of
%                       mean(Xc*bc + e)/std(Xc*bc + e)
%               if Xc hold only regressors of iC
%
%               NOTE: If you want to estimate the effect of a noise
%               correction using some regressors, the Contrast with index
%               iC should actually contain an F-contrast *EXCLUDING* these
%               regressors, because everything else constitutes the
%               regressors of interest
%   fileTsnr    tsnr_con<iC>.nii
%               path and file name of tSNR image, if doSave was true
%   tSnrRatioImage
%               tSNR_con<iC>./tSNR_con<iCForRatio>
%               (if iCForRatio was specified)
%   fileTsnrRatio
%               tsnr_con<iC>vs<iCRatio>.nii
%               file where tSnrRatio was saved
%
% EXAMPLE
%   compute_tsnr_spm
%
%   See also spm_write_residuals spm_FcUtil

% Author: Lars Kasper
% Created: 2014-12-10
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.


if nargin < 2
    iC = 0;
end

if nargin < 3
    iCForRatio = 0;
end

if nargin < 4
    doInvert = true;
end

if nargin < 5
    doSaveNewContrasts = false;
end


doComputeRatio = ~isempty(iCForRatio);


% load SPM-variable, if filename given
if ~isstruct(SPM)
    % load SPM variable from file
    if iscell(SPM)
        fileSpm = SPM{1};
    else
        fileSpm = SPM;
    end
    load(fileSpm, 'SPM');
end

oldDirSpm = SPM.swd;

if ~doSaveNewContrasts
    % temporary changes to SPM structure saved in sub-dir, removed later
    newDirSpm = fullfile(SPM.swd, 'tmp');
    mkdir(newDirSpm);
    copyfile(fullfile(SPM.swd, '*.nii'), newDirSpm);
    copyfile(fullfile(SPM.swd, 'SPM.mat'), newDirSpm);
    SPM.swd = newDirSpm;
end

iCIn = iC;

isInvertableContrast = iC > 0 && ~isnan(iC);

if isInvertableContrast && doInvert
    if ~isequal(SPM.xCon(iC).STAT, 'F')
        error('Can only invert F-contrasts');
    end
    
    idxColumnsContrast = find(sum(SPM.xCon(iC).c));
    
    Fc = spm_FcUtil('Set', ['All but: ' SPM.xCon(iC).name], 'F', ...
        'iX0', idxColumnsContrast, SPM.xX.xKXs);
    SPM.xCon(end+1) = Fc;
    SPM = spm_contrasts(SPM,numel(SPM.xCon));
    
    % use this for computation
    iC = numel(SPM.xCon);
end


%% Write residuals Y - Y0 = Yc + e;
VRes        = spm_write_residuals(SPM, iC);
nVolumes    = numel(VRes);
for iVol = 1:nVolumes
    VRes(iVol).fname = fullfile(SPM.swd, VRes(iVol).fname);
end

fileTsnr = fullfile(oldDirSpm, sprintf('tSNR_con%04d.nii', iCIn));

useMrImage = false;

if useMrImage % use toolbox functionality
    ResImage    = MrImage(VRes(1).fname);
    
    % Create 4D image of contrast-specific "residuals", i.e. Xc*bc + e
    for iVol = 1:nVolumes
        ResImage.append(VRes(iVol).fname);
    end
    
    % compute tSNR = mean(Xc*bc + e)/std(Xc*bc + e)
    tSnrImage = ResImage.mean./ResImage.std;
    if doSaveNewContrasts
        tSnrImage.save(fileTsnr);
    end
else
    ResImage = spm_read_vols(VRes);
    meanImage = mean(ResImage, 4);
    stdImage = std(ResImage, 0, 4);
    tSnrImage = meanImage./stdImage;
    VTsnr = VRes(1);
    VTsnr.fname = fileTsnr;
    spm_write_vol(VTsnr, tSnrImage);
end

%%
if doComputeRatio
    fileTsnrCompare = fullfile(oldDirSpm, sprintf('tSNR_con%04d.nii', iCForRatio));
    
    % Load or compute tSNR image for comparison contrast
    if ~exist(fileTsnrCompare, 'file');
        % when computed, don't compute another ratio, and don't delete tmp
        % here! (will be done at the end)
        tSnrCompareImage = tapas_physio_compute_tsnr_spm(...
            fullfile(oldDirSpm, 'SPM.mat'), ...
            iCForRatio, doInvert, [], 1);
    else
        VCompare = spm_vol(fileTsnrCompare);
        tSnrCompareImage = spm_read_vols(VCompare);
    end
    
    % compute tSNR ratio and save
    tSnrRatioImage = tSnrImage./tSnrCompareImage;
    
    fileTsnrRatio = fullfile(oldDirSpm, ...
        sprintf('tSNRRatio_con%04dvs%04d.nii', iCIn, iCForRatio));
    VRatio = spm_vol(fileTsnrCompare);
    VRatio.fname = fileTsnrRatio;
    spm_write_vol(VRatio, tSnrRatioImage);
else
    tSnrRatioImage = [];
    fileTsnrRatio = [];
end


%% clean up all created residual files and temporary SPM folder
if ~doSaveNewContrasts
    delete(fullfile(newDirSpm, '*'));
    rmdir(newDirSpm);
else
    % delete at least the Res-images
    delete(fullfile(oldDirSpm, 'Res_*.nii')); 
end