function this = MrAffineTransformation_constructor(this, testVariants)
% Unit test for MrAffineTransformation Constructor
%
%   Y = MrUnitTest()
%   run(Y, 'MrAffineTransformation_constructor')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrAffineTransformation_constructor
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2017-10-19
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% Unit test for MrAffineTransformation Constructor

switch testVariants
    case 'propVal' % test prop/val syntax
        
        % actual solution
        actSolution = this.make_affineTransformation_reference(0);
        
        % expected solution
        % get classes path
        classesPath = tapas_uniqc_get_path('classes');
        % make full filename
        solutionFileName = fullfile(classesPath, '@MrUnitTest' , 'affineTrafo.mat');
        expSolution = load(solutionFileName);
        expSolution = expSolution.affineTrafo;
        
        % verify equality of expected and actual solution
        % import matlab.unittests to apply tolerances for objects
        import matlab.unittest.TestCase
        import matlab.unittest.constraints.IsEqualTo
        import matlab.unittest.constraints.AbsoluteTolerance
        import matlab.unittest.constraints.PublicPropertyComparator
        
        this.verifyThat(actSolution, IsEqualTo(expSolution,...
            'Within', AbsoluteTolerance(10e-7),...
            'Using', PublicPropertyComparator.supportingAllValues));
        
    case 'matrix' % test affine transformation as input        
        % expected solution
        expSolution = this.make_affineTransformation_reference(0);
        expSolution = expSolution.affineMatrix;
        % actual solution
        % make actual solution from affine matrix of expected solution
        actSolution = MrAffineTransformation(expSolution);
        actSolution = actSolution.affineMatrix;
        
        % verify equality of expected and actual solution
        this.verifyEqual(actSolution, expSolution, 'absTol', 10e-7);
end
end