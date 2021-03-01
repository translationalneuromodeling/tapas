function this = MrAffineTransformation_affineMatrix(this)
% Unit test for MrAffineTransformation computing the affine Matrix
%
%   Y = MrUnitTest()
%   Y.MrAffineTransformation_affineMatrix(inputs)
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrAffineTransformation_affineMatrix
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


% construct MrAffineTransformation object from sampling points
affineTransformation = this.make_affineTransformation_reference(0);

% define expected solution
expSolution = affineTransformation;

% get and set affineMatrix
affineMatrix = affineTransformation.get_affine_matrix;
affineTransformation.update_from_affine_matrix(affineMatrix);

% define actual solution
actSolution = affineTransformation;

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
