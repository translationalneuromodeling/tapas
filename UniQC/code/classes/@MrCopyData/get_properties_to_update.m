function [sel, mobj] = get_properties_to_update(obj)
% Returns all properties which are mutable for update, cloning etc.
% (and the metaobject of property information)
%
%   Y = MrCopyData()
%   [sel, mobj] = Y.get_properties_to_update()
%
% This is a method of class MrCopyData.
%
% IN
%
% OUT
%
% EXAMPLE
%   get_properties_to_update
%
%   See also MrCopyData

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-23
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


mobj = metaclass(obj);

sel = find(cellfun(@(cProp)(~cProp.Constant && ...
    ~cProp.Abstract && ...
    (~cProp.Dependent || ...
    (cProp.Dependent && ...
    ~cProp.NonCopyable && ...
    ~isempty(cProp.SetMethod)))),mobj.Properties));
