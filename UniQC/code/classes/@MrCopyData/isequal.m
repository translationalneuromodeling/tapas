function isObjectEqual = isequal(obj, input_obj, ...
    tolerance)
% Sets all values of obj to [] which are the same in input_obj; i.e. keeps only the distinct differences in obj
%
%   Y = MrCopyData()
%   [diffObject, isObjectEqual] = isequal(obj, input_obj, ...
%    tolerance)
%
% This is a method of class MrCopyData.
%
% IN
% input_obj     the input MrCopyData from which common elements are subtracted
% tolerance     allowed difference seen still as equal
%               (default: eps(single)
%
% OUT
% diffObject    obj "minus" input_obj
% isObjectEqual true, if obj and input_obj were the same
%
% NOTE: empty values in obj, but not in input_obj remain empty,
% so are not "visible" as different. That's why
% obj.isequal(input_obj) delivers different results from
% input_obj.isequal(obj)
%
% EXAMPLE
%   isequal
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
    tolerance = eps('single'); % machine precision for the used data format
end

isObjectEqual = isequal(class(obj), class(input_obj)); % same class to begin with...

if ~isObjectEqual
    return 
end

[sel, mobj] = get_properties_to_update(obj);


% loop over all selected valid properties
for k = sel(:)'
    pname = mobj.Properties{k}.Name;
    if isa(obj.(pname), 'MrCopyData') %recursive comparison
        isSubobjectEqual = obj.(pname).isequal(input_obj.(pname), tolerance);
        isObjectEqual = isObjectEqual & isSubobjectEqual;
    else
        % cell of MrCopyData also treated
        if iscell(obj.(pname)) && iscell(input_obj.(pname)) ...
                && ~isempty(obj.(pname)) ...
                && isa(obj.(pname){1}, 'MrCopyData')
            for c = 1:min(length(obj.(pname)),length(input_obj.(pname)))
                isSubobjectEqual = obj.(pname){c}.isequal ...
                    (input_obj.(pname){c}, tolerance);
                isObjectEqual = isObjectEqual & isSubobjectEqual;
            end
        else % not MrCopyData
            if ~isempty(input_obj.(pname)) && ~isempty(obj.(pname))
                p = obj.(pname);
                ip = input_obj.(pname);
                
                if ~isnumeric(p) % compare cells, strings via isequal (no tolerance)
                    isPropertyEqual = tapas_uniqc_isequaltol(p,ip, tolerance);
                else % check vector/matrix (size) and equality with numerical tolerance
                    isPropertyEqual = prod(double(size(p)==size(ip)));
                    if isPropertyEqual
                        isPropertyEqual = ~any(abs(p(:)-ip(:))>tolerance);
                    end
                end
                isObjectEqual = isObjectEqual & isPropertyEqual;
            else % at least one prop is empty, if not both are, then test is wrong
                isObjectEqual = isObjectEqual & ...
                    (isempty(input_obj.(pname)) && isempty(obj.(pname)));
            end
            
        end
    end
end
end
