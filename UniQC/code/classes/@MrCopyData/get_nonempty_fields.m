function [fieldNames, fieldValues] = get_nonempty_fields(obj)
% Returns field names and values of all empty fields of MrCopyData-object
%
%   Y = MrCopyData()
%   Y.get_nonempty_fields(inputs)
%
% This is a method of class MrCopyData.
%
% IN
%
% OUT
%
% EXAMPLE
%   Y.get_nonempty_fields
%
%   See also MrCopyData

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-12-09
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


fieldNames = cell(0,1);
fieldValues = cell(0,1);


%% Find all properties of MrCopyData to be searched
[sel, mobj] = get_properties_to_update(obj);


%% Loop over all properties distinguishing between variables and MrCopyData
% -objects which are treated recursively

for k = sel(:)'
    pname = mobj.Properties{k}.Name;
    currProp = obj.(pname);
    
    if isa(currProp, 'MrCopyData') 
    %% Recursive operation on MrCopyData-property
        
        [fieldNamesTmp, fieldValuesTmp] = ...
            currProp.get_nonempty_fields();
        fieldNames = [fieldNames; fieldNamesTmp(:)];
        fieldValues = [fieldValues; fieldValuesTmp(:)];
        
    else
        %% Cell of MrCopyData also treated recursively for each cell element
        
        if iscell(currProp) ...
                && length(currProp) ...
                && isa(currProp{1}, 'MrCopyData')
            
            nCellProp = numel(currProp);
            currPropArray = currProp;
            for c = 1:nCellProp
                currProp = currPropArray{c};
                [fieldNamesTmp, fieldValuesTmp] = ...
                    currProp.get_nonempty_fields();
                fieldNames = [fieldNames; fieldNamesTmp(:)];
                fieldValues = [fieldValues; fieldValuesTmp(:)];
            end
            
        else 
        %% No cell of MrCopyData, no MrCopyData...any other property, therefore 
        % treat differently, and append output
        
            if ~isempty(currProp)
                fieldNamesTmp = {pname};
                fieldValuesTmp = {currProp};
                fieldNames = [fieldNames; fieldNamesTmp(:)];
                fieldValues = [fieldValues; fieldValuesTmp(:)];
            end
        end
    end
end

end