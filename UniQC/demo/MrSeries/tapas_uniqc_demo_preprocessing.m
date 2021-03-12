% Script preprocessing
% Example of fMRI preprocessing - input for model estimation demo
%
%  preprocessing
%
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-03
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


clear;
close all;
clc;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (1) Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file name
pathExamples        = tapas_uniqc_get_path('examples');
pathData            = fullfile(pathExamples, 'nifti', 'data_first_level');

fileFunctional      = fullfile(pathData, 'single_echo.nii');
fileStructural      = fullfile(pathData, 'anatomy.nii');

dirResults          = ['preprocessing' filesep];

% create MrSeries object
S = MrSeries(fileFunctional);
% remove first five samples
S.data = S.data.select('t', 6:S.data.dimInfo.nSamples('t'));
% set save path (pwd/dirResults)
S.parameters.save.path = tapas_uniqc_prefix_files(S.parameters.save.path, ...
    dirResults);
% check geometry
disp(S.data.geometry);
% add anatomy
S.anatomy.load(fileStructural, 'updateProperties', 'none');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (2) Realign
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check data first
S.compute_stat_images();
S.mean.plot('colorBar', 'on');
S.snr.plot('colorBar', 'on', 'displayRange', [0 80]);
S.data.plot('z', 24, 'sliceDimension', 't');
% looks good - now realing
S.realign();
% check data again
S.compute_stat_images();
S.mean.plot('colorBar', 'on');
S.snr.plot('colorBar', 'on', 'displayRange', [0 80]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (3) Coregister
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% anatomy --> mean
S.parameters.coregister.nameStationaryImage = 'mean';
S.parameters.coregister.nameTransformedImage = 'anatomy';
S.coregister();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (4) Segment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute tissue probability maps structural
S.parameters.compute_tissue_probability_maps.nameInputImage = 'anatomy';
S.compute_tissue_probability_maps();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (5) Compute Masks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we only want a grey matter mask
S.parameters.compute_masks.nameInputImages = 'tissueProbabilityMapGm';
S.parameters.compute_masks.nameTargetGeometry = 'mean';
S.compute_masks;
% check overlay
S.mean.plot('overlayImages', S.masks{1});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (6) Smooth
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% smooth with twice the voxel size
S.parameters.smooth.fwhmMillimeters = abs(S.data.geometry.resolution_mm) .* 2;
S.smooth;
% check data again
S.compute_stat_images();
S.mean.plot('colorBar', 'on');
S.snr.plot('colorBar', 'on', 'displayRange', [0 80]);