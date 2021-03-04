function this = MrImageGeometry_constructor(this, testVariants)
% Unit test for MrImageGeometry Constructor with different inputs.
%
%   Y = MrUnitTest()
%   run(Y, 'MrImageGeometry_constructor')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrImageGeometry_constructor
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
%


% reference objects
dimInfo = this.make_dimInfo_reference;
affineTrafo = this.make_affineTransformation_reference;
imageGeom = this.make_imageGeometry_reference;

switch testVariants
    case 'makeReference' % test whether reference is valid
        % actual solution
        actSolution = imageGeom;
        
        % expected solution
        % get classes path
        classesPath = tapas_uniqc_get_path('classes');
        % make full filename
        solutionFileName = fullfile(classesPath, '@MrUnitTest' , 'imageGeom.mat');
        expSolution = load(solutionFileName);
        expSolution = expSolution.imageGeom;
        
    case 'dimInfoAndaffineTransformation'
        % expected solution
        expSolution = imageGeom;
        
        % acutal solution
        actSolution = MrImageGeometry(affineTrafo, dimInfo);
        
    case 'matrix' % test affine matrix as input
        % expected solution
        expSolution = affineTrafo.get_affine_matrix();
        
        % actual solution
        % make actual solution from affine matrix of expected solution
        actSolution = MrImageGeometry(expSolution);
        actSolution = actSolution.get_affine_matrix();
        
    case 'dimInfo'
        % actual Solution
        dimInfo.resolutions({'x', 'y', 'z'}) = imageGeom.resolution_mm;
        actSolution = MrImageGeometry(dimInfo);
        
        % expected solution
        expSolution = imageGeom;
        % offcentre, rotation and shear will be lost
        expSolution.offcenter_mm = [dimInfo.samplingPoints{'x'}(1), ...
            dimInfo.samplingPoints{'y'}(1), dimInfo.samplingPoints{'z'}(1),];
        expSolution.rotation_deg = [0 0 0];
        expSolution.shear = [0 0 0];      

    case 'affineTransformation'
        % expected solution
        expSolution = imageGeom.get_affine_matrix();
        affineTrafo.update_from_affine_matrix(expSolution);
        % actual Solution
        actSolution = get_affine_matrix(MrImageGeometry(affineTrafo));
                
    case 'timing_info'
        % expected solution
        expSolution.s       = 0.65;
        expSolution.ms      = 0.65;
        expSolution.samples = 0;
        
        % TR is given in seconds
        geom = MrImageGeometry(dimInfo);
        actSolution.(dimInfo.t.units{1}) = geom.TR_s;
        
        % TR is given in ms
        dimInfo.set_dims('t', 'units', 'ms', 'resolutions', ...
            dimInfo.t.resolutions * 1000);
        geom = MrImageGeometry(dimInfo);
        actSolution.(dimInfo.t.units{1}) = geom.TR_s;
        
        % TR is given in samples
        dimInfo.set_dims('t', 'units', 'samples', 'resolutions', 1);
        geom = MrImageGeometry(dimInfo);
        actSolution.(dimInfo.t.units{1}) = geom.TR_s;
        
        
end
% verify equality of expected and actual solution

switch testVariants
    case 'makeReference'
        % verify equality of expected and actual solution
        % import matlab.unittests to apply tolerances for objects
        import matlab.unittest.TestCase
        import matlab.unittest.constraints.IsEqualTo
        import matlab.unittest.constraints.AbsoluteTolerance
        import matlab.unittest.constraints.PublicPropertyComparator
        
        this.verifyThat(actSolution, IsEqualTo(expSolution,...
            'Within', AbsoluteTolerance(10e-7),...
            'Using', PublicPropertyComparator.supportingAllValues));
        
    otherwise
        % verify equality of expected and actual solution
        this.verifyEqual(actSolution, ...
            expSolution, 'absTol', 10e-7);
        
end


end
