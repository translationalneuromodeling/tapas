function [nTestFailed, nTestTotal, testResults] = tapas_physio_test(level)
% Interface of PhysIO-Toolbox-specific test functions to global tapas_test
%
%   [nTestFailed, nTestTotal] = tapas_physio_test(level)
%
% IN
%   level       Depth of testing required
%                Approximate run time per level
%                    0:  around a minute 
%                        here: unit tests only
%                    1:  around 5 minutes (a coffee)
%                        here: matlab-only integration tests, no SPM integration
%                    2:  around an hour  (time for lunch)
%                        here: matlab AND SPM integration tests, takes less
%                        than 10 minutes
%                    3:  overnight       (time to freakout [deadline])
% OUT
%   nTestFailed     number of failed tests
%   nTestTotal      total number of executed tests
%   testResults     detailed test results object (matlab.unittest.TestResult)
%                   which may be queried to retrieve error messages or
%                   rerun tests
%
% EXAMPLE
%   tapas_physio_test(0)
%
%   See also tapas_test matlab.unittest.TestResult

% Author:   Lars Kasper
% Created:  2022-09-04
% Copyright (C) 2022 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

if nargin < 1
    level = 0;
end

tic

testResults = [];

% Returns an error message, if no example data found
tapas_physio_get_path_examples();

if level >= 0
    testResults = [testResults tapas_physio_run_unit_tests()];
else
    nTestTotal = 0;
    nTestFailed = 0;
    return
end

if level == 1
    % code adapted from run_integration_tests, but chooses only tests with
    % 'matlab' in the name (SPM GUI tests have SPM in the name)
    import matlab.unittest.TestSuite;
    
    pathTests   = fullfile(fileparts(mfilename('fullpath')), 'integration');
    suiteFolder = TestSuite.fromFolder(pathTests, ...
        'IncludingSubfolders', true, 'Name', '*matlab_only*');
    testResults = [testResults run(suiteFolder)];
end

if level >= 2
    testResults = [testResults tapas_physio_run_integration_tests()];
end

nTestTotal =  numel(testResults);
nTestFailed = sum([testResults.Failed]);

% pretty summary output
fprintf('\n\n\n\tTable of all executed PhysIO Tests:\n\n');
disp(testResults.table);

toc
