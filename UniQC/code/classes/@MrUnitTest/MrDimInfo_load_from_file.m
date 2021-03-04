function this = MrDimInfo_load_from_file(this, testFile)
% Unit test for MrDimInfo Constructor from file
%
%   Y = MrUnitTest()
%   run(Y, 'MrDimInfo_load_from_file')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDimInfo_load_from_file
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2017-11-03
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% Unit test for MrDimInfo Constructor loading from different example files

switch testFile
    case '3DNifti'
        % 3D Nifti
        % actual solution
        dataPath = tapas_uniqc_get_path('data');
        niftiFile3D = fullfile(dataPath, 'nifti', 'rest', 'meanfmri.nii');
        actSolution = MrDimInfo(niftiFile3D);
        % expected solution
        % get classes path
        classesPath = tapas_uniqc_get_path('classes');
        expSolution = load(fullfile(classesPath, '@MrUnitTest' , ...
            'dimInfo-meanfmri.mat'));
        expSolution = expSolution.dimInfo;
    case '4DNifti'
        % 4D Nifti
        % actual solution
        dataPath = tapas_uniqc_get_path('data');
        niftiFile4D = fullfile(dataPath, 'nifti', 'rest', 'fmri_short.nii');
        actSolution = MrDimInfo(niftiFile4D);
        % expected solution
        % get classes path
        classesPath = tapas_uniqc_get_path('classes');
        expSolution = load(fullfile(classesPath, '@MrUnitTest' , ...
            'dimInfo-fmri_short.mat'));
        expSolution = expSolution.dimInfo;
    case 'Folder'
        % several files in folder
        % actual solution
        dataPath = tapas_uniqc_get_path('data');
        niftiFolder= fullfile(dataPath, 'nifti', 'split', 'full');
        actSolution = MrDimInfo(niftiFolder);
        % expected solution
        % get classes path
        classesPath = tapas_uniqc_get_path('classes');
        expSolution = load(fullfile(classesPath, '@MrUnitTest' , ...
            'dimInfo-full.mat'));
        expSolution = expSolution.dimInfo;
    case 'ParRec'
        % par/rec data
        % actual solution
        dataPath = tapas_uniqc_get_path('data');
        % par/rec
        parRecFile = fullfile(dataPath, 'parrec', 'rest_feedback_7T', 'fmri1.par');
        actSolution = MrDimInfo(parRecFile);
        % expected solution
        % get classes path
        classesPath = tapas_uniqc_get_path('classes');
        expSolution = load(fullfile(classesPath, '@MrUnitTest' , ...
            'dimInfo-fmri1.mat'));
        expSolution = expSolution.dimInfo;
end

warning('off', 'MATLAB:structOnObject');
this.verifyEqual(struct(actSolution), struct(expSolution), 'absTol', 10e-7);
warning('on', 'MATLAB:structOnObject');

end