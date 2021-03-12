function this = MrImageGeometry_load_from_file(this, testFile)
% Unit test for MrImageGeometry Constructor from file
%
%   Y = MrUnitTest()
%   run(Y, 'MrImageGeometry_load_from_file')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrImageGeometry_load_from_file
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2018-01-17
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


% Unit test for MrImageGeometry Constructor loading from different example files

switch testFile
    case '3DNifti'
        % 3D Nifti
        % actual solution
        dataPath = tapas_uniqc_get_path('data');
        niftiFile3D = fullfile(dataPath, 'nifti', 'rest', 'meanfmri.nii');
        actSolution = MrImageGeometry(niftiFile3D);
        % expected solution
        % get classes path
        classesPath = tapas_uniqc_get_path('classes');
        expSolution = load(fullfile(classesPath, '@MrUnitTest' , ...
            'imageGeom-meanfmri.mat'));
        expSolution = expSolution.imageGeom;
    case '4DNifti'
        % 4D Nifti
        % actual solution
        dataPath = tapas_uniqc_get_path('data');
        niftiFile4D = fullfile(dataPath, 'nifti', 'rest', 'fmri_short.nii');
        actSolution = MrImageGeometry(niftiFile4D);
        % expected solution
        % get classes path
        classesPath = tapas_uniqc_get_path('classes');
        expSolution = load(fullfile(classesPath, '@MrUnitTest' , ...
            'imageGeom-fmri_short.mat'));
        expSolution = expSolution.imageGeom;
    case 'Folder'
        % several files in folder
        % actual solution
        dataPath = tapas_uniqc_get_path('data');
        niftiFolder= fullfile(dataPath, 'nifti', 'split', 'full');
        actSolution = MrImageGeometry(niftiFolder);
        % expected solution
        % get classes path
        classesPath = tapas_uniqc_get_path('classes');
        expSolution = load(fullfile(classesPath, '@MrUnitTest' , ...
            'imageGeom-full.mat'));
        expSolution = expSolution.imageGeom;
    case 'ParRec'
        % par/rec data
        % actual solution
        dataPath = tapas_uniqc_get_path('data');
        % par/rec
        parRecFile = fullfile(dataPath, 'parrec', 'rest_feedback_7T', 'fmri1.par');
        actSolution = MrImageGeometry(parRecFile);
        % expected solution
        % get classes path
        classesPath = tapas_uniqc_get_path('classes');
        expSolution = load(fullfile(classesPath, '@MrUnitTest' , ...
            'imageGeom-fmri.mat'));
        expSolution = expSolution.imageGeom;
end

% verify equality of expected and actual solution
% import matlab.unittests to apply tolerances for objects
import matlab.unittest.TestCase
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.AbsoluteTolerance
import matlab.unittest.constraints.PublicPropertyComparator

this.verifyThat(actSolution, IsEqualTo(expSolution,...
    'Within', AbsoluteTolerance(10e-7),...
    'Using', PublicPropertyComparator.supportingAllValues));

end