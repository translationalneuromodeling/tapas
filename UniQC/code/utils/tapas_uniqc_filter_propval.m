function filteredPropvalArray = tapas_uniqc_filter_propval(propvalArray, propStruct)
% returns property/value pairs where name matches property name of a
% structure/object
%
%   filteredPropvalArray = tapas_uniqc_filter_propval(propvalArray, propStruct)
%
% IN
%   propvalArray    cell(1, 2*nProperties) property name/values
%   propStruct      structure variable of named properties
%
% OUT
%   filteredPropvalArray
%
% EXAMPLE
%   tapas_uniqc_filter_propval
%
%   See also

% Author:   Lars Kasper
% Created:  2016-02-20
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if isstruct(propStruct) || isobject(propStruct)
    propertyNames = fieldnames(propStruct)';
else % already a cell
    propertyNames = propStruct;
end

nProperties = numel(propvalArray)/2;

filteredPropvalArray = {};

% check whether property exists in struct, and add it to output
%filtered propval, if it does
for p = 1:nProperties
    hasProperty = ~isempty(tapas_uniqc_find_string(propertyNames, propvalArray{2*p-1}));
    if hasProperty
        filteredPropvalArray = [filteredPropvalArray, ...
            propvalArray(2*p+[-1 0])];
    end
end
