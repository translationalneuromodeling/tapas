% Script demo_dim_info
% Exemplifies creation and usage of MrDimInfo class for retrieving and
% manipulating indices of multi-dimensional array
%
%  demo_dim_info
%
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-01-23
% Copyright (C) 2016 Institute for Biomedical Engineering
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
%% 1. Construct common dimInfo objects:
%   a) 4D EPI-fMRI array, with fixed TR
%   b) 5D multi-coil time series
%   c) 5D multi-echo time series
%   d) Create 5D multi-coil time series via nSamples and ranges
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  a) creates standard 4D dim info array from arraySize
% presets: units, dimLabels, resolutions
arraySize   = [64 50 33 100];
dimInfo     = MrDimInfo('nSamples', arraySize);
disp(dimInfo);

%% b) creates standard 5D dim info array from arraySize and resolutions
% presets of dimLabels, startingPoint = [1 1 1 1 1];
arraySize   = [64 50 33 100 8];
resolutions = [3 3 3 2.5 1];
units       = {'mm', 'mm', 'mm', 's', 'nil'};
dimInfo2    = MrDimInfo('nSamples', arraySize, 'resolutions', resolutions, ...
    'units', units);
disp(dimInfo2);
% no empty units allowed, will be overwritten by default!

%% c) creates standard 5D dim info array from arraySize, resolutions,
% startingPoints
% no presets
arraySize   = [64 50 33 8 3];
resolutions = [3 3 3 1 25];
units       = {'mm', 'mm', 'mm', 'nil', 'ms'};
dimLabels   = {'x', 'y', 'z', 'coil', 'echo_time'};
firstSamplingPoint = [-110, -110, -80, 0, 15];
dimInfo3    = MrDimInfo('nSamples', arraySize, 'resolutions', resolutions, ...
    'units', units, 'dimLabels', dimLabels, ...
    'firstSamplingPoint', firstSamplingPoint);
disp(dimInfo3.ranges);

%% d) Create 5D multi-coil time series via nSamples and ranges
% no presets, resolutions computed automatically
dimInfo4 = MrDimInfo(...
    'nSamples', [128 96 35 8 1000], ...
    'dimLabels', {'x', 'y', 'z', 'coil', 't'}, ...
    'units', {'mm', 'mm', 'mm', '', 's'}, ...
    'ranges', {[2 256], [2 192], [3 105], [1 8], [0 2497.5]});
disp(dimInfo4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get parameters of dimInfo via get_dims and dimInfo.'dimLabel'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check out how neatly you can retrieve information about certain
% dimensions:
% dimInfo2.get_dims('x').resolutions
dimInfo2.x.resolutions
dimInfo2.get_dims('x')
dimInfo2.get_dims({'x' 'y' 'z'})

%% even cooler: use dimLabels directly for referencing!
dimInfo2.z.samplingPoints
dimInfo2.nSamples('z')
dimInfo2.nSamples({'z','y'})

dimInfo2('z')
dimInfo2({'z', 'y'})
dimInfo2([3 2])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Modify dimInfo-dimensions via set_dims/add_dims-command
% a) Specify non-consecutive sampling-points (e.g. coil channels)
% b) Shift start sample of dimensions (e.g. centre FOV in x/y)
% c) Add a 6th dimension (e.g. additinal echoes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  a) Specify non-consecutive sampling-points (e.g. coil channels)
disp(dimInfo3);
dimInfo3.set_dims('coil', 'samplingPoints', [2 3 4 7 8 10 11 12])
% Note that there is no concept of resolution here anymore, since there is
% equidistant spacing!
dimInfo3.resolutions
% However, samplingWidths is retained, since the data is still coming from
% one coil (a better way to think of it would be to select a number of
% slices; the resolution would be nan but each slice would still have a
% thickness = samplingWidth [which is the true slice thickness, resolution
% would be composed of samplingWidth + sliceGap])
dimInfo3.samplingWidths


% b) Shift start sample of dimensions (e.g. centre FOV in x/y)
disp(dimInfo4);
dimInfo4.set_dims([1 2], 'arrayIndex', [65 49], 'samplingPoint', [0 0]);
dimInfo4.ranges(:,1:2)

disp(dimInfo3);
dimInfo3.samplingPoints('coil') = {13:15};
dimInfo3.z.samplingPoints = {1:20};
dimInfo3.echo_time.nSamples = 5;

% c) Add a 6th dimension (e.g. additinal echoes)
disp(dimInfo2);
dimInfo2.add_dims(6, 'samplingPoints', [17 32], 'dimLabels', 'echo', 'units', 'ms');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Display sampling points (=absolute indices with units) of
% selected first/center/last voxel
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

arrayIndexFirst     = [1,1,1,1,1];
arrayIndexLast      = [128 96 35 8 1000];
arrayIndexCenter    = [64, 48, 18, 4, 500];

arrayIndices = [
    arrayIndexFirst
    arrayIndexCenter
    arrayIndexLast
    ];

nVoxels = size(arrayIndices, 1);

samplingPointArray = dimInfo4.index2sample(...
    arrayIndices);

fprintf('===\ndimInfo.sample2index(arrayIndices): \n');
for iVoxel = 1:nVoxels
    fprintf('array Index, voxel %d:', iVoxel);
    fprintf('%5.1f  ', samplingPointArray(iVoxel,:));
    fprintf('\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Display strings with units: sampling point of selected
%   first/center/last voxel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


indexLabelArray = dimInfo4.index2label(...
    arrayIndices);

fprintf('===\ndimInfo.index2label(arrayIndices): \n');
for iVoxel = 1:nVoxels
    fprintf('Voxel %d: ', iVoxel);
    fprintf('%s ',indexLabelArray{iVoxel}{:});
    fprintf('\n\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Back-transform: Retrieve voxel index from absolute index
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

retrievedVoxelIndexArray = dimInfo4.sample2index(samplingPointArray);

fprintf('===\ndimInfo.sample2index(samplingPointArray): \n');
for iVoxel = 1:nVoxels
    fprintf('retrieved array index, voxel %d:', iVoxel);
    disp(retrievedVoxelIndexArray(iVoxel,:));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. dimInfo.select() - extract subset of dimension info from
%                        PropName/Value-pairs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  a) Select only data from a specific subset of one dimension (e.g. all
% data from some coils)
[selectionDimInfo, selectionIndexArray] = dimInfo4.select('type', 'index', ...
    'coil', [2 3 5 6]);

%% b) Select all data excluding subsets of one dimension (e.g. first volumes
% of time series data)
[selectionDimInfo2, selectionIndexArray2] = dimInfo4.select('type', 'index', ...
    'invert', true, 't', [1:5]);

%% c) Select subset of data array from multiple dimensions (e.g. cubic volume ROI,
% but see MrRoi for more sophisticated region definitions)
[selectionDimInfo3, selectionIndexArray3] = dimInfo4.select('type', 'index', ...
    'x', [20:40 80:100],  'y', [1:16 81:96], 'z', [10:15]);

%% d) Select subset of data array by specifying sampling-points
[selectionDimInfo4, selectionIndexArray4] = dimInfo4.select('type', 'sample', ...
    'x', [-128:2:0],  'y', [0:2:80]);

%% e) Combine selection into a nice structure, instead of
% ParameterName/Value-pairs (not all have to be given!)
selection.x = 1:10;
selection.t = 200:300;
[selectionDimInfo5, selectionIndexArray5] = dimInfo4.select(selection);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. dimInfo = MrDimInfo(fileName) - extract dimInfo directly from file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  3D Nifti
dataPath = tapas_uniqc_get_path('data');
niftiFile3D = fullfile(dataPath, 'nifti', 'rest', 'meanfmri.nii');
dimInfo3DFile = MrDimInfo(niftiFile3D);

% 4D Nifti
niftiFile4D = fullfile(dataPath, 'nifti', 'rest', 'fmri_short.nii');
dimInfo4DFile = MrDimInfo(niftiFile4D);

% 5D nifti
niftiFile5D = fullfile(dataPath, 'nifti', '5D', 'y_5d_deformation_field.nii');
dimInfo5DFile = MrDimInfo(niftiFile5D);

% several files in folder
niftiFolder = fullfile(dataPath, 'nifti', 'split', 'full');
dimInfoFolder = MrDimInfo(niftiFolder);

% par/rec
parRecFile = fullfile(dataPath, 'parrec', 'rest_feedback_7T', 'fmri1.par');
dimInfoParRec = MrDimInfo(parRecFile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 6. Create MrDimInfo object from dimInfo struct (MrDimInfo(dimInfoStruct))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% creat MrDimInfo object
dimInfoObject = MrDimInfo('nSamples', [64 50 33 100], ...
    'units', {'u1', 'u2', 'u3', 'u4'}, ...
    'dimLabels', {'l1', 'l2', 'l3', 'l4'}, ...
    'resolutions', [0.3 2 0.85 5]);
disp(dimInfoObject);
% convert to struct (gives a warning)
dimInfoStruct = struct(dimInfoObject);
disp(dimInfoStruct);
% now create a new dimInfo object using the struct as input
newDimInfoObject = MrDimInfo(dimInfoStruct);









    