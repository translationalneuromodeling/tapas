% Script demo_constructor
% Shows versatile options for creating MrImage object
%
%  demo_constructor
%
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-03-01
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
%% Load from workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nSamples = [48, 48, 5, 3, 8, 7];
data = randn(nSamples);
data = tapas_uniqc_create_image_with_index_imprint(data);
I = MrImage(data, ...
    'dimLabels', {'x', 'y', 'z', 't', 'echo', 'coil'}, ...
    'units', {'mm', 'mm', 'mm', 's', 'ms', 'nil'}, ...
    'resolutions', [1.5 1.5 1 0.5 17 1]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load from filename
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% Path Settings

pathExamples    = tapas_uniqc_get_path('data');
fileTestArray        = {
    fullfile(pathExamples, 'nifti', 'rest', 'meanfmri.nii') ...
    fullfile(pathExamples, 'nifti', 'rest', 'fmri_short.nii') ...
    fullfile(pathExamples, 'nifti', '5D', 'y_5d_deformation_field.nii') ...
    fullfile(pathExamples, 'nifti', 'split', 'subset', '*fmri*.nii') ...
    fullfile(pathExamples, 'nifti', 'split', 'subset') ...
    fullfile(pathExamples, 'parrec', 'rest_feedback_7T', 'fmri1.par')
    };

% load 3D nifti file
disp(['loading file: ', fileTestArray{1}]);
Img{1} = MrImage(fileTestArray{1});
Img{1}.plot; disp(Img{1}.geometry);

%% load 4D nifti file
disp(['loading file: ', fileTestArray{2}]);
Img{2} = MrImage(fileTestArray{2});
Img{2}.plot; disp(Img{2}.geometry);

%% load 5D nifti file
disp(['loading file: ', fileTestArray{3}]);
Img{3} = MrImage(fileTestArray{3}, 'dimLabels', ...
    {'x', 'y', 'z', 't', 'deformationField'}, 'units', ...
    {'mm', 'mm', 'mm', 'samples', 'nil'});
Img{3}.plot; disp(Img{3}.dimInfo);

%% load multiple nifti files using wildcard
disp(['loading file: ', fileTestArray{4}]);
Img{4} = MrImage(fileTestArray{4});
Img{4}.plot; disp(Img{4}.dimInfo);

%% load multiple files from folder
disp(['loading file: ', fileTestArray{5}]);
Img{5} = MrImage(fileTestArray{5});
Img{5}.plot; disp(Img{5}.dimInfo);

%% load par/rec file
disp(['loading file: ', fileTestArray{6}]);
Img{6} = MrImage(fileTestArray{6});
Img{6}.plot; disp(Img{6}.geometry);




