function this = MrDimInfo_get_add_remove(this)
% Unit test for MrDimInfo get, add and remove method
%
%   Y = MrUnitTest()
%   run(Y, 'MrDimInfo_get_add_remove')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDimInfo_methods
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
%


% construct MrDimInfo object from sampling points
dimInfo = this.make_dimInfo_reference(0);

% define expected solution
expSolution = dimInfo.copyobj;

% get, remove and add dims

% get dimInfo along x
dimInfoX = dimInfo.get_dims('x');

% remove x from dimInfo
dimInfo.remove_dims('x');

% add x back to dimInfo
dimInfo.add_dims(1, 'samplingPoints', dimInfoX.samplingPoints{1}, ...
    'dimLabels', dimInfoX.dimLabels{1}, 'units', dimInfoX.units{1});

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

