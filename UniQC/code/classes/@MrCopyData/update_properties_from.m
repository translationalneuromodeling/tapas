function update_properties_from(obj, input_obj, overwrite)
% Updates properties of obj for all non-empty values of inputpobj recursively ...
%
%   Y = MrCopyData()
%   Y.update_properties_from(input_obj, overwrite)
%
% This is a method of class MrCopyData.
%
% IN
% input_obj     either MrCopyData or a struct with the same sub-structures
%               (properties) as the object
% overwrite     0, {1}
%               0 = don't overwrite set values in obj; set empty values to set values in input_obj
%               1 = overwrite all values in obj, which have non-empty values in input_obj;
%               2 = overwrite all values
%
% OUT
% obj           updated obj w/ properties of input-obj
%
% EXAMPLE
%   update_properties_from
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


if nargin < 3
    overwrite = 1;
end

[sel, mobj] = obj.get_properties_to_update();

for k = sel(:)'
    pname = mobj.Properties{k}.Name;
    % check whether input_obj has prop of equivalent name
    % or, if input_obj is a struct, field of that name (the 2 queries are
    % not equivalent!)
    if isfield(input_obj, pname) || isprop(input_obj, pname)
        if isa(obj.(pname), 'MrCopyData') %recursive comparison
            obj.(pname).update_properties_from ...
                (input_obj.(pname), overwrite);
        else
            % cell of MrCopyData also treated
            if iscell(obj.(pname)) && iscell(input_obj.(pname)) ...
                    && length(obj.(pname)) ...
                    && isa(obj.(pname){1}, 'MrCopyData')
                for c = 1:min(length(obj.(pname)),length(input_obj.(pname)))
                    obj.(pname){c}.update_properties_from ...
                        (input_obj.(pname){c}, overwrite);
                    
                end
            end
            if (overwrite == 2) || ...
                    (~isempty(input_obj.(pname)) && (isempty(obj.(pname)) || overwrite))
                obj.(pname) = input_obj.(pname);
            end
        end
    end
end
end