% Script demo_load_fileformats - to be excluded
% Shows versatile file format loading capabilities of MrImage.load
%
% demo_load_fileformats
%
%
% See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-02-14
% Copyright (C) 2017 Institute for Biomedical Engineering
% University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
% <http://www.gnu.org/licenses/>.

pathExamples    = tapas_uniqc_get_path('data');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Load different types of nifti
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a) load data from fileName and updates both name and parameters.save of
% nifti files, header is read to update MrImage.parameters
fileName = fullfile(pathExamples, 'nifti', 'rest', 'fmri_short.nii');

Y1 = MrImage(fileName);
disp(Y1);
disp(Y1.dimInfo);

%% b) load data from fileName with different load options

% selectedVolumes: Note that selectedVolumes is specific to loading nifti,
% cpx and par/rec files and refers to the 4rd dimension of the loaded data;
% the advantage is it avoids loading the full data and then selecting a subset.
Y2 = MrImage(fileName, 'selectedVolumes', 3:2:9);
disp(Y2.dimInfo);

% updateProperties: Note that per default the name of the MrImage object is
% set to the file. For more information see MrDataNd.read_single_file. In
% the example here, the save parameters are set to the load path. Per
% default, a new path depending on the pwd is created to prevent accidental
% overwrite. Be careful with this option!
Y3 = MrImage(fileName, 'updateProperties', 'save');
disp(Y3.parameters.save);

%% c) load data from fileName with additinal dimInfo information

% default: The dimInfo information is gathered from the header (same as Y1,
% nothing to do here).
Y4 = MrImage(fileName);

% fileName_dimInfo.mat: An additional dimInfo.mat file exisits.
Y4.dimInfo.units = {'this', 'and', 'that', 'too'};
Y4.save;

% load file, dimInfo is automatically added
Y5 = MrImage(Y4.get_filename);
disp(Y5.dimInfo);

% compare to only loading the nifti file
[fp, fn] = fileparts(Y4.get_filename);
delete(fullfile(fp, [fn, '_dimInfo.mat']));
Y6 = MrImage(Y4.get_filename);
disp(Y6.dimInfo);

% restore correct dimInfo via dimInfo argument
Y7 = MrImage(Y4.get_filename, 'dimInfo', Y5.dimInfo);

disp(Y7.dimInfo);

% or, alternatively, via prop/val pair
Y8 = MrImage(Y4.get_filename, 'units', Y4.dimInfo.units);
disp(Y8.dimInfo)

%% d) load filename from data with additional affine geometry
Y9 = MrImage(fileName, 'shear', [0.2 -0.1 0]);
disp(Y9.geometry);

%% 2. Load multiple files in folder
% a) load multiple .nii files in folder with filenames containing additional
% dimension information but no additinal _dimInfo.mat files
fileNameSplitSubset = fullfile(pathExamples, 'nifti', 'split', 'subset');
YSplitSubset = MrImage(fileNameSplitSubset);
disp(YSplitSubset.dimInfo);

% b) load multiple nifti files in folder without the filenames containing
% dimension information
fileNameSplitRes = fullfile(pathExamples, 'nifti', 'split', 'split_residual_images');
YSplitRes = MrImage(fileNameSplitRes);
disp(YSplitRes.dimInfo);

% c) load multiple nifti files in folder with filenames containing
% additional dimension information and select
fileNameSplitFull = fullfile(pathExamples, 'nifti', 'split', 'subset');
select.TE = 2;
select.t = 3;
YSplitSelect = MrImage(fileNameSplitFull, 'select', select);

% d) load multiple nifti files in folder with filenames containing
% additional dimension information and _dimInfo file
fileNameSplitDimInfo = fullfile(pathExamples, 'nifti', 'split', 'with_dim_info');
YSplitDimInfo = MrImage(fileNameSplitDimInfo);

% e) load multiple nifti files in folder with filenames containing
% additional dimension information and select which is not a dimension of
% the files
selectError.TE = 2;
selectError.t = 7;
selectError.doesNotExist = 3;
YSplitError = MrImage(fileNameSplitFull, 'select', selectError);

% f) select for MrSeries
YMrSeries = MrSeries(fileName, 'select', {'t', 4:2:8});
disp(YMrSeries.data.dimInfo);