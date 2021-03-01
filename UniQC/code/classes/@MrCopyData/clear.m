function clear(obj)
% Recursively sets all non-MrCopyData-objects to empty([])
% to clear default values
%
%   Y = MrCopyData()
%   Y.clear(inputs)
%
% This is a method of class MrCopyData.
%
% IN
%
% OUT
%
% EXAMPLE
%   clear
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

[sel, mobj] = get_properties_to_update(obj);
for k = sel(:)'
    pname = mobj.Properties{k}.Name;
    if isa(obj.(pname), 'MrCopyData') %recursive comparison
        obj.(pname).clear;
    else
        % cell of MrCopyData also treated
        if iscell(obj.(pname)) ...
                && length(obj.(pname)) ...
                && isa(obj.(pname){1}, 'MrCopyData')
            for c = 1:length(obj.(pname))
                obj.(pname){c}.clear;
            end
        else
            obj.(pname) = [];
        end
    end
end
end