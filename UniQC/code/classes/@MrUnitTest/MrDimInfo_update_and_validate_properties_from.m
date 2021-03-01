function this = MrDimInfo_update_and_validate_properties_from(this)
% Unit test for MrDimInfo methods update_and_validate_properties_from
%
%   Y = MrUnitTest()
%   run(Y, 'MrDimInfo_update_and_validate_properties_from')
%   Y.MrDimInfo_update_and_validate_properties_from(TestParameter)
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDimInfo_update_and_validate_properties_from
%
%   See also MrUnitTest

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-07-04
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

dimInfo = this.make_dimInfo_reference(0);

% specify new properties
samplingPoints5D = ...
    {1.5:111, [2,4,6,7], 1.5:111, -24:0.1:-10, 0:2.123:54};
% create dimInfo object
expSolution = MrDimInfo(...
    'dimLabels', {'L1', 'L2', 'L3', 'L4', 'L5'}, ...
    'units', {'U1', 'U2', 'U3', 'U4', 'U5'}, ...
    'samplingPoints', samplingPoints5D);
% creat input struct for update (remove samplingPoints field)
input = expSolution.get_struct();
input = rmfield(input, {'samplingPoints', 'samplingWidths'});

% create actual solution
actSolution = dimInfo.copyobj();
actSolution.update_and_validate_properties_from(expSolution);

% verify whether expected and actual solution are identical
% Note: convert to struct, since the PublicPropertyComparator (to allow
% nans to be treated as equal) does not compare properties of objects that
% overload subsref

warning('off', 'MATLAB:structOnObject');
this.verifyEqual(struct(actSolution), struct(expSolution), 'absTol', 10e-7);
warning('on', 'MATLAB:structOnObject');

end