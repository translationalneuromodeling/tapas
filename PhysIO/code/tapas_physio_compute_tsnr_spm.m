function tSnrImage = tapas_physio_compute_tsnr_spm(SPM, iC)
% Computes temporal SNR image after correcting for a contrast 
% from SPM general linear model
%
%   tSnrImage = compute_tsnr_spm(input)
%
% IN
%   SPM     SPM variable (in SPM.mat, or file name) after parameter and 
%           contrast estimation
%   iC      Contrast of interest for which tSNR is computed
%
%           iC = 0 computes tSNR of the raw image time series after
%               pre-whitening and filtering, but without any model
%               confound regressors removed. (default)
%           iC = NaN computes tSNR after removing the full model (including
%               the mean), and is therefore equivalent to sqrt(1/ResMS)
%            
%   
% OUT
%   tSnrImage   MrImage holding tSNR when only including regressors in
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
%
% EXAMPLE
%   compute_tsnr_spm
%
%   See also spm_write_residuals spm_FcUtil
%
% Author: Lars Kasper
% Created: 2014-12-10
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.
% $Id: tapas_physio_compute_tsnr_spm.m 782 2015-07-23 15:05:28Z kasperla $

if nargin < 2
    iC = 0;
end

% load SPM-variable, if filename given
if ~isstruct(SPM)
    % load SPM variable from file
    if iscell(SPM)
        fileSpm = SPM{1};
    else
        fileSpm = SPM;
    end
    SPM = load(fileSpm, 'SPM');
end

% Write residuals Y - Y0 = Yc + e;
VRes        = spm_write_residuals(SPM, iC);
nVolumes    = numel(VRes);
ResImage    = MrImage(fullfile(SPM.swd, VRes(1).fname));

% Create 4D image of contrast-specific "residuals", i.e. Xc*bc + e
for iVol = 1:nVolumes
    ResImage.append(fullfile(SPM.swd, VRes(iVol).fname));
end
    
% compute tSNR = mean(Xc*bc + e)/std(Xc*bc + e)
tSnrImage = ResImage.mean./ResImage.std;