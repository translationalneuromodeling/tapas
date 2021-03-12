function this = MrAffineTransformation_transformation(this)
% Unit test for MrAffineTransformation applying transformations
%
%   Y = MrUnitTest()
%   run(Y, 'MrAffineTransformation_transformation')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrAffineTransformation_transformation
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2018-01-16
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
expSolution = affineTransformation.copyobj;

% create transformation matrix
transformationMatrix = [-0.2769 0.6948 0.4387 18.69; ...
    0.0462 -0.3171 0.3816 4.898; ...
    0.0971 0.9502 -0.7655 4.456; ...
    -0.8235 0.0344 0.7952 0.6463];
% apply transformation matrix
affineTransformation.apply_transformation(transformationMatrix);
% verify that affine transformation is applied
if affineTransformation.isequal(expSolution)
    this.assertFail('The transformation matrix has not been applied.');
end
% apply inverse transformation matrix
affineTransformation.apply_inverse_transformation(transformationMatrix);

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
