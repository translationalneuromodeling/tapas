function this = update_and_validate_properties_from(this, dimInfo)
% Updates properties from other dimInfo, allowing only valid entries to
% update. Can update from MrDimInfo object or struct.
%
%   Y = MrDimInfo()
%   Y.update_and_validate_properties_from(dimInfo)
%
% This is a method of class MrDimInfo.
%
% IN
%
% OUT
%
% EXAMPLE
%   update_and_validate_properties_from
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-24
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% Check whether valid input has been supplied.
dimInfoIsObject = isobject(dimInfo);
dimInfoIsStruct = isstruct(dimInfo);

if ~(dimInfoIsObject || dimInfoIsStruct)
    error('tapas:uniqc:MrDimInfoInvalidInputObjectStruct', ...
        'Invalid input supplied. Only MrDimInfo objects or matching structs');
else
    tempDimInfoArgs = [];
    % get the number of properties/fields supplied
    if dimInfoIsObject
        dimInfoProperties = properties(dimInfo);
    else
        dimInfoProperties = fieldnames(dimInfo);
    end
    nArgs = numel(dimInfoProperties);
    
    % only do this here if dimInfo object - allows for trailing singleton
    % dimensions
    if dimInfoIsObject
        % check nDims
        if isequal(this.nDims, dimInfo.nDims) % check whether nDims of given
            % dimInfo matches this
            % if yes, nothing to do here
            
        elseif isequal(1:this.nDims, dimInfo.get_non_singleton_dimensions) % check
            % whether non singleton dimensions match, i.e. whether there are any
            % singleton dimensions at the end
            
            % if yes, add singelton dimensions to this to enable comparison
            singletonDimensions = dimInfo.get_singleton_dimensions;
            this.add_dims(singletonDimensions, ...
                'dimLabels', dimInfo.dimLabels(singletonDimensions));
        else % else, error
            error('tapas:uniqc:MrDimInfoUpdateNonMatchingNumberOfDimensions', ...
                'Number of dimensions in input dimInfo does not match current nDims');
        end
    end
    for iArg = 1:nArgs
        currProp = dimInfoProperties{iArg};
        % make sure current property is not nDims (no set)
        isnDims = strcmp(currProp, 'nDims');
        if ~isnDims
            % extract current value
            currVal = dimInfo.(currProp);
            % no empty or nan properties used, no zeros for nSamples used
            isnSamples = strcmp(currProp, 'nSamples');
            if iscell(currVal)
                if strcmp(currProp, 'dimLabels')
                    % check whether any difference between current and old
                    % values for dimLabels
                    oldVal = this.(currProp);
                    if dimInfoIsObject || (dimInfoIsStruct && (numel(oldVal) == numel(currVal)))
                        isValidProperty = any(~(cellfun(@isequal, currVal, oldVal)));
                    else
                        isValidProperty = 0;
                    end
                elseif strcmp(currProp, 'units')
                    % check whether any difference between current and old values
                    % for units
                    % do this of dimInfo is object
                    % OR dimInfo is struct AND dimensions match
                    oldVal = this.(currProp);
                    if dimInfoIsObject || (dimInfoIsStruct && (numel(oldVal) == numel(currVal)))
                        isValidProperty = any(~(cellfun(@isequal, currVal, oldVal)));
                    else
                        isValidProperty = 0;
                    end
                elseif ismember(currProp, {'samplingPoints', 'samplingWidths'})
                    oldVal = this.(currProp);
                    if dimInfoIsObject || (dimInfoIsStruct && (numel(oldVal) == numel(currVal)))
                        % check whether nans or empty values were given
                        isNans = cellfun(@(C) any(isnan(C(:))), currVal);
                        isEmpty = cellfun(@(C) any(isempty(C(:))), currVal);
                        isValidProperty = ~all(isNans) && ~all(isEmpty);
                    else
                        isValidProperty = 0;
                    end
                end
            else
                oldVal = this.(currProp);
                isValidProperty = ~all(isnan(currVal(:))) && ~all(isempty(currVal(:))) && ...
                    ~(isnSamples && all(currVal(:) == 0));
            end
            % compare nDims here for struct
            if isValidProperty && dimInfoIsStruct
                if ~(numel(oldVal) == numel(currVal))
                    error('tapas:uniqc:MrDimInfoUpdateNonMatchingNumberOfDimensions', ...
                        'Number of dimensions in input dimInfo does not match current nDims');
                end
            end
            
            if isValidProperty
                tempDimInfoArgs = [tempDimInfoArgs {currProp} {currVal}];
            end
        end
    end
    if ~isempty(tempDimInfoArgs)
        this.set_dims(1:this.nDims, tempDimInfoArgs{:});
    end
    
end

end