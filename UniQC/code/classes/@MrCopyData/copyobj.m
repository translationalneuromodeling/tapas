function new = copyobj(obj, varargin)
% This method acts as a copy constructor for all derived classes.
%
% new = obj.copyobj('exclude', {'prop1', 'prop2'});
%
% IN
%   'exclude'           followed by cell(nProps,1) of property
%                       names that should not be copied
%                       NOTE: Properties of Class CopyObj are
%                       always copied, even if they have a
%                       name listed in this array%
%
% EXAMPLE
%   copyobj
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


defaults.exclude = {''};
args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);
exclude = cellstr(exclude);


new = feval(class(obj)); % create new object of correct subclass.

% Only copy properties which are
% * not dependent or dependent and have a SetMethod
% * not constant
% * not abstract
% * not nonCopyable
% * defined in this class or have public SetAccess - not
% sure whether this restriction is necessary
[sel, mobj] = get_properties_to_update(obj);


for k = sel(:)'
    pname = mobj.Properties{k}.Name;
    if isa(obj.(pname), 'MrCopyData') %recursive deep copy
        new.(pname) = ...
            obj.(pname).copyobj('exclude', exclude);
    else
        isPropCell = iscell(obj.(pname));
        if isPropCell ...
                && ~isempty(obj.(pname)) ...
                && isa(obj.(pname){1}, 'MrCopyData')
            new.(pname) = cell(size(obj.(pname)));
            for c = 1:length(obj.(pname))
                new.(pname){c} = ...
                    obj.(pname){c}.copyobj('exclude', exclude);
            end
        else
            % if matrix named data, don't copy
            isExcludedProp = ismember(pname, exclude);
            if isExcludedProp
                if isPropCell
                    new.(pname) = {};
                else
                    new.(pname) = [];
                end
            else
                new.(pname) = obj.(pname);
            end
            
        end
    end
end
end