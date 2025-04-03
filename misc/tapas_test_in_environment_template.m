function worked = tapas_test_in_environment(logfile,toolbox)
% TAPAS_TEST_IN_ENVIRONMENT run test in separate environment
%
% IN 
%   logfile [character]
%       Logfile to print output to.
%   toolbox [character]
%       Toolbox to test. If empty, all toolboxes will be tested.
% 
% Authors: Matthias Müller-Schrader & Lars Kasper
% Created: 2023-05-08
% Copyright (C) 2023 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
%

% NOTE: This file has a different filename than the function it defines.
% The reason is that it will be copied by run_tapas_test_in_environment 
% to a file with the proper name (to avoid shadowing of the path).

if nargin < 1 || isempty(logfile)
    logfile = 'tapas-test.log'; % Will be in temp-folder.
end
diary(logfile); % Write output to file.
if nargin < 2 || isempty(toolbox)
   disp('Testing all tooboxes!')
    toolbox = {}; % Everything.
elseif ~iscell(toolbox)
    toolbox = {toolbox};
end

fprintf('Running tapas_test_in_environment at %s\n',datetime)
%fprintf('MATLAB-Version:\n\t%s\n',version())
ver(); % Use not recommended anymore, but gives very nice overview over installed toolboxes
restoredefaultpath; % Ensure that path is clean.
addpath(fullfile(pwd,'spm')); % Add SPM-folder.
cd('tapas'); % Go into tapas-folder to init tapas
tapas_init(toolbox{:}); % Init tapas.
cd('..') % Execute rest in temporal folder.
tapas_download_example_data(); % Download example data for the tests.
nFailed = tapas_test(toolbox{:}); % Run the tests.
diary('off')
exit(min(nFailed,255)); % Exit > 0: We had a problem.
