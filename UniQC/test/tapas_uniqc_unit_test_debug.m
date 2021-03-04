% Script unit_test_debug
% Examples how to perform individual unit tests for debugging. Tags or the
% unit test name can be used. Note that changes in the class definitions
% requires a clear classes to take effect.
%
%  unit_test_debug
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
 
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run test for MrDimInfo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create test suite
import matlab.unittest.TestSuite;

UTDimInfo = TestSuite.fromClass(?MrUnitTest,'Tag','MrDimInfo');
resultsDimInfo = run(UTDimInfo);
disp(table(resultsDimInfo));

% run individual test for MrDimInfo
% create test object
testCase = MrUnitTest;
% call individual test cases
res = run(testCase, 'MrDimInfo_constructor');
res = run(testCase, 'MrDimInfo_get_add_remove');
res = run(testCase, 'MrDimInfo_split');
res = run(testCase, 'MrDimInfo_select');
res = run(testCase, 'MrDimInfo_load_from_file');
res = run(testCase, 'MrDimInfo_load_from_mat');
res = run(testCase, 'MrDimInfo_permute');
res = run(testCase, 'MrDimInfo_update_and_validate_properties_from')
testCase.MrDimInfo_constructor('11');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run test for MrAffineTransformation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UTaffineTransformation = TestSuite.fromClass(?MrUnitTest,'Tag','MrAffineTransformation');
resultsaffineTransformation = run(UTaffineTransformation);
disp(table(resultsaffineTransformation));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run test for MrImageGeometry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UTImageGeometry = TestSuite.fromClass(?MrUnitTest,'Tag','MrImageGeometry');
resultsImageGeometry = run(UTImageGeometry);
disp(table(resultsImageGeometry));

% call individual test cases
res = run(testCase, 'MrImageGeometry_constructor');
res = run(testCase, 'MrImageGeometry_load_from_file');
res = run(testCase, 'MrImageGeometry_create_empty_image');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run test for MrDataNd
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UTDataNd = TestSuite.fromClass(?MrUnitTest,'Tag','MrDataNd');
resultsDataNd = run(UTDataNd);
disp(table(resultsDataNd));

res = run(testCase, 'MrDataNd_select');
res = run(testCase, 'MrDataNd_arithmetic_operation');
res = run(testCase, 'MrDataNd_value_operation');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run test for MrImage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
res = run(testCase, 'MrImage_load_from_file'); 
testCase.MrImage_load_from_file('FilePlusOriginIndex');