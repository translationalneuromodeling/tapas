function varargout = tapas_physio_run_unit_tests()
% Executes all existing unit tests of PhysIO Toolbox
% NOTE: Unit tests are meant to test small, but critical nuclei of the
%       code, such as numerical routines (convolution, PCA...)
%       For more complex workflows (e.g. whole example scripts), use
%       integration tests
%
%   testResults = tapas_physio_run_unit_tests()
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_run_unit_tests
%
%   See also tapas_physio_run_integration_tests
 
% Author:   Lars Kasper
% Created:  2019-03-14
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
 
import matlab.unittest.TestSuite;

pathTests = fullfile(fileparts(mfilename('fullpath')), 'unit');
suiteFolder = TestSuite.fromFolder(pathTests, 'IncludingSubfolders', true);
testResults = run(suiteFolder);

% pretty summary output
disp(testResults.table);

if nargout
    varargout{1} = testResults;
end