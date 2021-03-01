% Script notesUnitTest
% ONE_LINE_DESCRIPTION
%
%  notesUnitTest
%
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-11-30
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% How to use tolerances with objects in unit testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 does not work for objects that overwrite subsref

import matlab.unittest.constraints.AbsoluteTolerance
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.PublicPropertyComparator
% for more info see matlab.unittest.constraints.IsEqualTo class

expSolution = makeExpectedSolution;
actSolution = makeActualSolution;

verifyThat(this, ...
    actSolution, ...
    IsEqualTo(expSolution, 'Within', AbsoluteTolerance(1e-6), ...
    'Using', PublicPropertyComparator.supportingAllValues));