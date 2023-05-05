function [nTestFailed,nTestTotal] = tapas_test(toolboxes,level)
% TAPAS_TEST Run tests of TAPAS toolboxes
%
% IN
%   toolboxes
%           Names of toolboxes to test (including their dependencies)-
%           If empty, all toolboxes are used.
%   level
%           Detail of the test (per toolbox).
%           0: Roughly a minute.
%           1: Coffebreak.
%           2: Lunch break.
%           3: Overnight. 
%   
% OUT
%   
%
% This function serves as a wrapper function to run the tests of the TAPAS toolboxes.
%
% See also 
%
% (C) 18.07.22, Translational Neuromodeling Unit, UZH & ETH Zürich, Matthias Müller-Schrader

doTestAll = false;

if nargin < 1 || isempty(toolboxes)
    toolboxes = {};
    doTestAll = true;
else
    toolboxes = lower(toolboxes);
    if ~iscell(toolboxes)
	toolboxes = {toolboxes}; % We want to have it as a cell array afterwards.
    end
end

if nargin < 2 || isempty(level)
    level = 3;
end

 % In the infos struct, we have all information about our toolboxes.

infos = tapas_get_toolbox_infos();
% Coding here is similar to tapas_init_toolboxes
toolbox_names = fieldnames(infos);
nTool = numel(toolbox_names); % All tapas-toolboxes
NTool = numel(toolboxes); % All specified toolboxes
if doTestAll
    doTest = ones(size(toolbox_names),'logical');
else
    doTest = zeros(size(toolbox_names),'logical');
    for iTool = 1:NTool % Run over specified toolboxes 
        sTool = toolboxes{iTool};
        if ~ismember(sTool,toolbox_names)
        	warning('I do not know the toolbox %s - skipping it.\n',sTool);
        	continue; 
        end
        doTest(ismember(toolbox_names,sTool)) = true;
        dependencies = lower(infos.(sTool).dependencies);
        if ~isempty(dependencies) 
        	doTest(ismember(toolbox_names,dependencies)) = true;
        end
    end
end

% Now run through the tests of the required toolboxes.
testResults = zeros(nTool,2);
for iTool = 1:nTool
    if ~doTest(iTool)
        continue;
    end
    toolboxName = toolbox_names{iTool};
    testFunctionName = infos.(toolboxName).test_function_name;
   
    str = sprintf('~~~~~~~~~~~~~~~~~~~~~~~~ TESTING  <strong>%s</strong> ~',...
                upper(toolboxName));
    str(end+1:80+17) = '~'; % 17 for <strong></strong>
    fprintf(1,'%s\n',str);
    if isempty(testFunctionName)
        fprintf('Tests not implemented yet.\n')
        %fprintf('\tTest for toolbox %s not implemented yet.\n',toolboxName)
        continue;
    end
    hTestFunction = str2func(testFunctionName);
    
    try 
        [nTestFailed,nTestTotal] = hTestFunction(level);
    catch me
        warning('Test for toolbox failed with message:\n\t%s',me.message);%#ok
        nTestFailed = 1;
        nTestTotal = 1;
    end
    testResults(iTool,1) = nTestFailed;
    testResults(iTool,2) = nTestTotal;
    fprintf('<strong>Summary</strong> %s: %d out of %d tests failed.\n',...
        toolboxName,nTestFailed,...
        nTestTotal)
end

nTestFailed = sum(testResults(:,1));
nTestTotal = sum(testResults(:,2));
fprintf(['~~~~~~~~~~~~~~~~~~~~~~~~ <strong>SUMMARY</strong> ~~~~~~~~~~~~~~~~~~~~',...
    '~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'])
fprintf('<strong>Summary</strong>: %d of %d tests failed\n',nTestFailed,nTestTotal);

end

function [nTestFailed,nTestTotal] = tapas_test_template(level)
    % This minimal working example demonstrates the interface
    % for the toolbox-specific test-functions. 
    %
    % Approximate time per level (per toolbox)
    %   0:  around a minute
    %   1:  around 5 minutes (a coffee)
    %   2:  around an hour  (time for lunch)
    %   3:  overnight       (time to freakout [deadline])
    error('Hallloooo')
    if level < 1
        nTestTotal = 2;
    else
        nTestTotal = 3;
    end
    nTestFailed = 0;
    randomNumber = randn(1);
    if randomNumber < 0
        fprintf('First test failed :(\n')
        nTestFailed = nTestFailed + 1;
    end

    randomNumberUnitInterval = rand();
    if randomNumberUnitInterval < 0.5
        fprintf('Second test failed :(\n')
        nTestFailed = nTestFailed + 1;
    end

    if level >= 1
        pause(300); % To justify the level ;)
        randomInteger = randi(10,1);
        if randomInteger < 5
            fprintf('Third test failed :( \n')
            nTestFailed = nTestFailed + 1;
        end
    end

end


