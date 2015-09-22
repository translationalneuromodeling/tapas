function varargout = ...
    tapas_physio_compute_tsnr_gains(physio, SPM, doSave)
% Computes tSNR gains through physiological noise correction for all
% confound regressor sets modelled, after estimation of the SPM-GLM
%
%   [fileTsnrGainArray, fileTsnrArray] = ...
%           tapas_physio_compute_tsnr_gains(physio, SPM, doSave);
%
%   This function executes the following steps:
%   1.  Compute F-Contrasts of the kind "All-but-physiological confounds"
%       to estimate tSNR from spm-residuals
%       NOTE: If contrasts exist, they are not re-created, but used as is
%   2.  Compute tSNR of preprocessed time-series, after pre-whitening and
%       highpass-filtering (i.e. K*W*Y), saved as nii-files
%       using tapas_physio_compute_tsnr_spm(SPM, 0);
%   3.  Compute tSNR images of individual physiological regressors sets,
%       using F-contrasts from (1.), saved as nii-files
%   4.  Compute tSNR ratio images from (3.) and (2.), save as nii-files
%
% IN
%   physio      physio-structure (or filename with structure), after estimating multiple_regressors.txt
%   SPM         SPM variable (or filename with saved structure)
%               after estimation of general linear model (GLM)
% OUT
%   fileTsnrGainArray
%               cell(nPhysioSets,1) of nii-filenames holding tSNR gain
%               images for all physiological regressor sets (in percent?)
%
%   fileTsnrArray
%               cell(nPhysioSets+1,1) of nii-filenames for tSNR images of
%               for all physiological regressor sets
%               and, as last element, raw tSNR (after preprocessing)
% EXAMPLE
%   tapas_physio_compute_tsnr_gains
%
%   See also tapas_physio_compute_tsnr_spm spm_write_residuals
%
% Author: Lars Kasper
% Created: 2015-07-03
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_compute_tsnr_gains.m 785 2015-07-31 11:00:04Z kasperla $



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   0. Check variable structure, cast for convenience
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    doSave = true;
end

% load physio-variable, if filename given
if ~isstruct(physio)
    % load SPM variable from file
    if iscell(physio)
        filePhysio = physio{1};
    else
        filePhysio = physio;
    end
    load(filePhysio, 'physio');
end

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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   1.  Compute F-Contrasts of the kind "All-but-physiological confounds"
%       to estimate tSNR from spm-residuals
%       NOTE: If contrasts exist, they are not re-created, but used as is
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Determine which physiological contrasts (should) exist and where the
% corresponding columns in the design matrix are

[colAll, colCard, colResp, colMult, colHRV, colRVT, colMove] = ...
    tapas_physio_check_get_regressor_columns(SPM, physio.model);

con{1} = colAll;
con{2} = colCard;
con{3} = colResp;
con{4} = colMult;
con{5} = colHRV;
con{6} = colRVT;
con{7} = colMove;

namesPhysContrasts = {
    'All Phys'
    'Cardiac'
    'Respiratory'
    'Card X Resp Interation'
    'HeartRateVariability'
    'RespiratoryVolumePerTime'
    'Movement'
    };

iEmptyCon                       = cellfun(@isempty, con);

con(iEmptyCon)                  = [];
namesPhysContrasts(iEmptyCon)   = [];
nContrasts                      = numel(con);

% indices of newly created inverse contrasts, i.e. all columns but
% physiological ones...

indInverseContrasts = zeros(nContrasts,1);
for iC = 1:nContrasts
    Fc = spm_FcUtil('Set', ['All but: ' namesPhysContrasts{iC}], 'F', ...
        'iX0', con{iC}, SPM.xX.xKXs);
    SPM.xCon(end+1) = Fc;
    SPM = spm_contrasts(SPM,length(SPM.xCon));
    
    indInverseContrasts(iC) = numel(SPM.xCon);
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   2.  Compute tSNR of preprocessed time-series, after pre-whitening and
%       highpass-filtering (i.e. K*W*Y), saved as nii-files
%       using tapas_physio_compute_tsnr_spm(SPM, 0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tSnrImageRaw = tapas_physio_compute_tsnr_spm(SPM, 0);
tSnrImageRaw.name = 'raw tSNR after preprocessing, pre-whitening, high-pass filtering';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   3.  Compute tSNR images of individual physiological regressors sets,
%       using F-contrasts from (1.), saved as nii-files
%   AND
%   4.  Compute tSNR ratio images from (3.) and (2.), save as nii-files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TODO use spm_imcalc for calculation and spm_read_vols/spm_write_vol for
% input/output of images
% for all computed Ics...
tSnrImageArray = cell(nContrasts+1,1);
tSnrGainArray = cell(nContrasts,1);
for iC = 1:nContrasts
    tSnrImageArray{iC} = tapas_physio_compute_tsnr_spm(SPM, ...
        indInverseContrasts(iC));
    tSnrImageArray{iC}.name = ['tSNR after correcting ' ...
        namesPhysContrasts{iC}];
    tSnrImageArray{iC}.parameters.save.path = SPM.swd;
    tSnrImageArray{iC}.parameters.save.fileName = ...
        sprintf('tSnr_%s.nii', namesPhysContrasts{iC});
    tSnrGainArray{iC} = (tSnrImageArray{iC}./tSnrImageRaw - 1).*100;
    tSnrGainArray{iC}.name = ['tSNR gain after correcting ' ...
        namesPhysContrasts{iC}];
    tSnrGainArray{iC}.parameters.save.path = SPM.swd;
    tSnrGainArray{iC}.parameters.save.fileName = ...
        sprintf('tSnrGain_%s.nii', namesPhysContrasts{iC});
end

tSnrImageArray{end} = tSnrImageRaw;
tSnrImageArray{end}.parameters.save.path = SPM.swd;
tSnrImageArray{end}.parameters.save.fileName = ...
    sprintf('tSnr_Raw.nii');

if doSave
    for iC = 1:nContrasts
        tSnrImageArray{iC}.save;
        tSnrGainArray{iC}.save;
    end
    tSnrImageArray{end}.save;
end

doPlot = false;
if doPlot
   spm_check_registration(tSnrImageArray{end}.parameters.save.fileName);
end

if nargout
    varargout{1} =  tSnrGainArray;
    
    if nargout > 1
        varargout{2} = tSnrImageArray;
    end
end