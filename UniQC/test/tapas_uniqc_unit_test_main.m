% Script unit_test_main
% Performs the complete unit test for uniQC. All unit test specified in
% MrUnitTest are included.
%
%  unit_test_main
%
%
%   See also MrUnitTest
 
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-04-04
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% create test suite
import matlab.unittest.TestSuite;
% run complete test suit
UTall = TestSuite.fromClass(?MrUnitTest);
resultsAll = run(UTall);
disp(table(resultsAll));

