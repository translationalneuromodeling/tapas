function this = MrDataNd_constructor(this, testVariantsDataNd)
% Unit test for MrDataNd constructor with different input variants
%
%   Y = MrUnitTest()
%   run(Y, 'MrDataNd_constructor');
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDataNd_constructor
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2018-02-08
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


% get the dimInfo reference
dimInfo = this.make_dimInfo_reference(0);
% change some parameters to check defaults are overwritten
dimInfo.dimLabels = {'dL1', 'dL2', 'dL3', 'dL4', 'dL5'};
dimInfo.units = {'u1', 'u2', 'u3', 'u4', 'u5'};
% remove samples to make data matrix smaller
dimInfo.dL4.nSamples = 10;
rng(1);
dataMatrix = rand(dimInfo.nSamples);


switch testVariantsDataNd
    case 'matrix'
        % load matrix from workspace
        actSolution = MrDataNd(dataMatrix);
        expSolution = actSolution;
        
    case 'matrixWithDimInfo'
        % load matrix with dimInfo object
        actSolution = MrDataNd(dataMatrix, 'dimInfo', dimInfo);
        expSolution = actSolution;
    case 'matrixWithPropVal'
        % load matrix with prop/val pairs describing dimInfo
        actSolution = MrDataNd(dataMatrix, 'dimLabels', dimInfo.dimLabels, ...
            'units', dimInfo.units, 'samplingPoints', dimInfo.samplingPoints);
        expSolution = actSolution;
end
% verify whether expected and actual solution are identical
% Note: convert dimInfo to struct, since the PublicPropertyComparator (to allow
% nans to be treated as equal) does not compare properties of objects that
% overload subsref

warning('off', 'MATLAB:structOnObject');
actSolution.dimInfo = struct(actSolution.dimInfo);
warning('on', 'MATLAB:structOnObject');
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


