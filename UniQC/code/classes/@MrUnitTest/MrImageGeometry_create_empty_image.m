function this = MrImageGeometry_create_empty_image(this)
% Unit test for MrImageGeometry.create_empty_image
%
%   Y = MrUnitTest()
%   run(Y, 'MrImageGeometry_create_empty_image')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrImageGeometry_create_empty_image
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2018-01-18
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% expected solution
imageGeom = this.make_imageGeometry_reference;
expSolution = imageGeom;

% actual solution
emptyImage = imageGeom.create_empty_image;
actSolution = emptyImage.geometry;

% verify equality of expected and actual solution
% import matlab.unittests to apply tolerances for objects
import matlab.unittest.TestCase
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.AbsoluteTolerance
import matlab.unittest.constraints.PublicPropertyComparator

this.verifyThat(actSolution, IsEqualTo(expSolution,...
    'Within', AbsoluteTolerance(10e-7),...
    'Using', PublicPropertyComparator.supportingAllValues));


