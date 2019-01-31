function indexFoundPattern = tapas_physio_find_string(stringArray, searchPattern, isExact)
% Finds a string pattern in a cell of strings and returns found cell indices
%
% NOTE: this function behaves the same as find should behave
%
%   indexFoundPattern = find_string(stringArray, searchPattern, isExact)
%
% IN
%       stringArray     cell(nStrings,1) of cells to be checked for search
%                       pattern
%       searchPattern   string to be searched for (regular expressions
%                       possible
%                       OR cell array of search patterns
%       isExact         default:false; if true, only exact matched of
%                       searchPattern to strings are returned
% OUT
%       indexFoundPattern vector of indices in stringArray that match
%                       searchPattern
%                       OR, if searchPattern was cell array:
%                       cell(nPatterns,1) of index vectors, one cell
%                       element for each pattern
% EXAMPLE
%   find_string
%
%   See also

% Author: Lars Kasper
% Created: 2014-11-05
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.

if ~iscell(stringArray)
    stringArray = {stringArray};
end

if nargin < 3
    isExact = false;
end


if ~iscell(searchPattern)
    
    if isExact
        % enforce start and end of string match that in sought-through
        searchPattern = ['^' searchPattern '$'];
    end
    
    
    indexFoundPattern = find(~cell2mat(cellfun(@isempty, regexp(stringArray, ...
        searchPattern), ...
        'UniformOutput', false)));
else
    
    nPatterns = numel(searchPattern);
    indexFoundPattern = cell(nPatterns,1);
    for iPattern = 1:nPatterns
        
        if isExact
            % enforce start and end of string match that in sought-through
            searchPattern{iPattern} = ['^' searchPattern{iPattern} '$'];
        end
        
        
        indexFoundPattern{iPattern} = find(~cell2mat(cellfun(@isempty, regexp(stringArray, ...
            searchPattern{iPattern}), ...
            'UniformOutput', false)));
    end
end

end