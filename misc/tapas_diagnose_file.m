function [fList,isFine] = tapas_diagnose_file(fileNames,filterNames,notifyMultipleOccurences)
% TAPAS_DIAGNOSE_FILE Checks if file uses functions which are overloaded
%
% IN
%   fileName [character vector of cell array of character vectors]
%           The file names to be checked. 
%
% OUT
%   fList   Filtered list of files needed to execute the function
%   isFine  No problem found.
%
% This function is using the matlab codetools to determine 
% which functions are used and warns, if some of these are
% overshadowed. This might take a while (a few minutes); 
% mostly due to the call of codetools which is performing
% recursive checks on all/most functions involved.
%
% See also 
%
% (C) 21.01.22, Translational Neuromodeling Unit, UZH & ETH Zürich, Matthias Müller-Schrader

if nargin < 2 || isempty(filterNames)
    filterNames = { strcat(filesep,'SPM12',filesep) ,strcat(filesep,'tapas_')};
end
if nargin < 3 || isempty(notifyMultipleOccurences)
    notifyMultipleOccurences = true;
end


if ~iscell(fileNames)
    fileNames = {fileNames}; % For printing function names
end

%% Check that functions are in the path
for iFunction = 1:numel(fileNames)
    if isempty(which(fileNames{iFunction}))
        error('Function %s is not contained in the path.',fileNames{iFunction})
    end
end

%% Use matlab.codetools.requiredFilesAndProducts to find dependencies
% This method does not return build-in functions, but functions overshadowing them

fprintf('Diagnosing dependencies for file(s):\n')
fprintf('\t%s\n',fileNames{:})
tic()
[fList, pList] = matlab.codetools.requiredFilesAndProducts(fileNames);
%% Filter out  some cases (SPM and tapas functions)
toFilter = zeros(size(fList),'logical'); % indices of function to filter out
for iFilter = 1:numel(filterNames)
    toFilter = toFilter | contains(fList,filterNames{iFilter});
end
fList = fList(~toFilter);
nCheck = numel(fList);
fprintf('Check for %d dependencies (%d found /%d filtered) in %0.1fs.\n',nCheck,numel(toFilter),numel(toFilter)-nCheck,toc())
isFine = true;
%% For the remaining functions: check whether they are overshadowed
for iCheck = 1:nCheck % loop through files
    [~,funName] = fileparts(fList{iCheck});
    % The information regarding shadowed is unfortunately only in the console output
    % of which - not in the returned string. Therefore evalc is used.
    whichCommand = sprintf('which(''%s'',''-all'')',funName);
    whichListExtended = evalc(whichCommand);
    if contains(whichListExtended,'% Shadowed')
        warning('Function %s is shadowed',funName)
        disp(whichListExtended)
        isFine = false;
    elseif notifyMultipleOccurences
        nRes = numel(strfind(whichListExtended,newline));
        if nRes > 1
            fprintf('Found %d occurences for %s but that is probably fine.\n',nRes,funName);
        end
    end
end

if isFine
    fprintf('Did not find a problem for the file(s):\n')
    fprintf('\t%s\n',fileNames{:})
end
