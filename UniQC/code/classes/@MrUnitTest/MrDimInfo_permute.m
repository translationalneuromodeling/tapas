function this = MrDimInfo_permute(this)
% Unit test for MrDimInfo permute
%
%   Y = MrUnitTest()
%   run(Y, 'MrDimInfo_permute')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDimInfo_permute
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2018-01-15
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% construct MrDimInfo object from sampling points
dimInfo = this.make_dimInfo_reference(0);

% define permutation
permutation = [3 1 4 2 5];

% create expected solution
samplingPointsPermuted = dimInfo.samplingPoints(permutation);
dimLabelsPermuted = dimInfo.dimLabels(permutation);
unitsPermuted = dimInfo.units(permutation);

expSolution = MrDimInfo('samplingPoints', samplingPointsPermuted, ...
    'dimLabels', dimLabelsPermuted, 'units', unitsPermuted);

% permute
dimInfo.permute([3 1 4 2 5]);

% define actual solution
actSolution = dimInfo;

% verify whether expected and actual solution are identical
% Note: convert to struct, since the PublicPropertyComparator (to allow
% nans to be treated as equal) does not compare properties of objects that
% overload subsref

warning('off', 'MATLAB:structOnObject');
this.verifyEqual(struct(actSolution), struct(expSolution), 'absTol', 10e-7);
warning('on', 'MATLAB:structOnObject');

end
