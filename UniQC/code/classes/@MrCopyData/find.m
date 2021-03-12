function foundHandles = find(obj, nameClass, varargin)
% Finds all (handle) object properties whose class and property
% values match given values
%
%   foundHandles = obj.find(nameClass, 'PropertyName',
%                           'PropertyValue', ...)
%
%
% IN
%   nameClass       string with class name (default: MrCopyData)
%   PropertyName/   pairs of string containing name of property
%   PropertyValue   and value (or pattern) that has to be matched
%                   NOTE: if cells of values are given, all
%                   objects are returned that match any of the
%                   entries,
%                   e.g. 'name', {'mean', 'snr'} will return
%                   objects if they are named 'mean OR 'snr'
%
% OUT
%   foundHandles    cell(nHandles,1) of all object handles for
%                   objects that match the properties
%
%                   NOTE: function also returns handle to
%                   calling object, if its class and properties
%                   fulfill the given criteria
%
% EXAMPLE:
%   Y = MrCopyData();
%   Y.find('MrCopyData', 'name', 'coolCopy');
%
%   See also MrCopyData

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-04-19
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


foundHandles = {};
if nargin
    searchPropertyNames = varargin(1:2:end);
    searchPropertyValues = varargin(2:2:end);
    nSearchProperties = numel(searchPropertyNames);
    
else
    nSearchProperties = 0;
end

% check whether object itself fulfills criteria
if isa(obj, nameClass)
    
    doesMatchProperties = true;
    iSearchProperty = 1;
    
    % Check search properties as long as matching to values
    while doesMatchProperties && ...
            iSearchProperty <= nSearchProperties
        
        searchProperty = searchPropertyNames{iSearchProperty};
        searchValue = searchPropertyValues{iSearchProperty};
        if isa(obj.(searchProperty), 'MrCopyData')
            % recursive comparison for MrCopyData-properties
            doesMatchProperties = obj.(searchProperty).comp(...
                searchValue);
        else
            
            doesMatchProperties = isequal(obj.(searchProperty), ...
                searchValue);
            
            % allow pattern matching for strings or cell of
            % strings (matching any entry of cell)
            if ischar(obj.(searchProperty))
                cellSearchValue = cellstr(searchValue);
                nCellEntries = numel(cellSearchValue);
                iCellEntry = 1;
                
                % check for each entry in cell whether it
                % matches the string value of this object's
                % property
                while iCellEntry <= nCellEntries && ~doesMatchProperties
                    currentSearchValue = ...
                        cellSearchValue{iCellEntry};
                    
                    % check whether pattern expression given, i.e.
                    % * in search value
                    isSearchPattern = ~isempty(strfind(currentSearchValue, ...
                        '*'));
                    if isSearchPattern
                        doesMatchProperties = ~isempty(regexp( ...
                            obj.(searchProperty), currentSearchValue, 'once'));
                    else
                        doesMatchProperties = isequal(obj.(searchProperty), ...
                            currentSearchValue);
                    end
                    iCellEntry = iCellEntry + 1;
                end
            end
            
        end
        
        iSearchProperty = iSearchProperty + 1;
    end
    
    if doesMatchProperties
        foundHandles = [foundHandles; {obj}];
    end
end

[sel, mobj] = get_properties_to_update(obj);
for k = sel(:)'
    pname = mobj.Properties{k}.Name;
    
    % Continue to check properties recursively for MrCopyData-properties
    if isa(obj.(pname), 'MrCopyData') % recursive comparison
        newFoundHandles = obj.(pname).find(nameClass, varargin{:});
        foundHandles = [foundHandles;newFoundHandles];
    else
        % cell of MrCopyData also treated
        if iscell(obj.(pname)) && length(obj.(pname)) ...
                && isa(obj.(pname){1}, 'MrCopyData')
            for c = 1:length(obj.(pname))
                newFoundHandles = obj.(pname){c}.find(nameClass, varargin{:});
                foundHandles = [foundHandles;newFoundHandles];
            end
        end
    end
end
% remove duplicate entries, if sub-object itself was returned and
% as a property of super-object
% foundHandles = unique(foundHandles);
end