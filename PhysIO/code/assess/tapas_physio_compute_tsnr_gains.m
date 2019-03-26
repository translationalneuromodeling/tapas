function [tSnrImageArray, fileTsnrArray, ...
            tSnrRatioImageArray, fileTsnrRatioArray ] = ...
    tapas_physio_compute_tsnr_gains(physio, SPM, indexContrastForSnrRatio, ...
    namesPhysContrasts)
% Computes tSNR gains through physiological noise correction for all
% confound regressor sets modelled, after estimation of the SPM-GLM
%
%   [fileTsnrGainArray, fileTsnrArray] = ...
%           tapas_physio_compute_tsnr_gains(physio, SPM, doSave);
%
%
% IN
%   physio      physio-structure (or filename with structure), after estimating multiple_regressors.txt
%   SPM         SPM variable (or filename with saved structure)
%               after estimation of general linear model (GLM)
%   indexContrastForSnrRatio
%               contrast id for comparison to get relative tSNR values
%               default: 0 (raw time series after filtering/pre-whitening)
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

% Author: Lars Kasper
% Created: 2015-07-03
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   0. Check variable structure, cast for convenience
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    indexContrastForSnrRatio = 0; % contrast for comparison
end

if nargin < 4
    namesPhysContrasts = tapas_physio_get_contrast_names_default();
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
% 1. Determine which physiological contrasts could be created with existing
% model and create the missing ones
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SPM = tapas_physio_create_missing_physio_contrasts(SPM, physio.model, ...
    namesPhysContrasts);

nContrasts = numel(namesPhysContrasts);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Compute tSNR maps via tapas_physio_compute_tsnr_spm, as follows
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tSnrImageArray = cell(nContrasts,1);
fileTsnrArray = cell(nContrasts,1);
tSnrRatioImageArray = cell(nContrasts,1);
fileTsnrRatioArray = cell(nContrasts,1);
            
for c = 1:nContrasts
    % First try, whether inverted contrast already exists, then no
    % additional computation has to be performed!
    indContrast = tapas_physio_check_get_xcon_index(SPM, ...
        ['All but: ' namesPhysContrasts{c}]);
    doComputeInvertedContrast = indContrast == 0; % not found, do invert
    doSaveNewContrasts = doComputeInvertedContrast; % and save contrasts for later
    if doComputeInvertedContrast % find contrast before inversion
        indContrast = tapas_physio_check_get_xcon_index(SPM, ...
            namesPhysContrasts{c});
    end
    
    if indContrast > 0
        % if contrast exist, compute tSNR and save it!
        [tSnrImageArray{c}, fileTsnrArray{c}, ...
            tSnrRatioImageArray{c}, fileTsnrRatioArray{c}] = ...
            tapas_physio_compute_tsnr_spm(SPM, indContrast, ...
            indexContrastForSnrRatio, doComputeInvertedContrast, doSaveNewContrasts);
        
        % rename files for better human readability
        filenameSfx = regexprep(namesPhysContrasts{c}, ' ', '_');
        fileTsnrOld = fileTsnrArray{c};
        fileTsnrArray{c} = regexprep(fileTsnrArray{c}, ...
            sprintf('con%04d', indContrast), filenameSfx);
        
        fileTsnrRatioOld = fileTsnrRatioArray{c};
        fileTsnrRatioArray{c} = regexprep(fileTsnrRatioArray{c}, ...
            sprintf('con%04dvs%04d', indContrast, indexContrastForSnrRatio), ...
            [filenameSfx '_vs_Raw']);
        
        movefile(fileTsnrOld, fileTsnrArray{c});
        movefile(fileTsnrRatioOld, fileTsnrRatioArray{c});
    end
end