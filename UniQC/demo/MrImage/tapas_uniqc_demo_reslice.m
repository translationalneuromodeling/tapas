% Script demo_reslice
% ONE_LINE_DESCRIPTION
%
%  demo_reslice
%
%
%   See also
 
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-11-27
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
 
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0. Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathData        = tapas_uniqc_get_path('examples');

fileFunctional  = fullfile(pathData, 'nifti', 'rest', 'fmri_short.nii');
fileStructural      = fullfile(pathData, 'nifti', 'rest', 'struct.nii');

% stationary image is the mean functional
EPI = MrImage(fileFunctional);
EPI.parameters.save.fileName = 'funct.nii';
% moving image is the structural
anat = MrImage(fileStructural);
anat.parameters.save.fileName = 'struct.nii';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Reslice structural image to geometry of functional image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EPI.plot;
anat.plot;
rAnat = anat.reslice(EPI.mean);
rAnat.plot;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Reslice 4D time series
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% default: each volume of the 4D time series is resliced
rEPI = EPI.reslice(anat);
rEPI.plot('t', rEPI.dimInfo.nSamples('t'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Reslice with different SPM parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% interpolation (but only first volume to speed up the computation)
rAnatInterpolation = anat.reslice(EPI.mean, 'interpolation', 2);
plot(rAnat - rAnatInterpolation);
% wrapping left-right
rAnatWrapping = anat.reslice(EPI.mean, 'wrapping', [0 0 1]);
plot(rAnat - rAnatWrapping);
% masking of missing voxel in time series
EPI.data(:,:,26,6:EPI.dimInfo.nSamples) = nan;
% with masking
rEPIWithMasking = EPI.reslice(anat);
rEPIWithMasking.plot();
% without masking
rEPIWithoutMasking = EPI.reslice(anat, 'masking', 0);
rEPIWithoutMasking.plot('t', 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Reslice 5D multi-echo fMRI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathExamples = tapas_uniqc_get_path('examples');
pathMultiEcho = fullfile(pathExamples, 'nifti', 'data_multi_echo');

% loads all 4D nifti files (one per echo) in 5D array; takes dim name of
% 5th dimension from file name
ME = MrImage(fullfile(pathMultiEcho, 'multi_echo*.nii'));

TE = [9.9, 27.67 45.44];
ME.dimInfo.set_dims('echo', 'units', 'ms', 'samplingPoints', TE);

rME = ME.reslice(EPI.mean.geometry);
rME.plot('t', 1);