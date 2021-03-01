function [isObjectEqual, out_left, out_right] = comp(...
    obj, input_obj, tolerance)
% Returns non-empty properties where  obj and input_obj differ ...
%
%
%   Y = MrCopyData()
%   Y.comp(inputs)
%
% This is a method of class MrCopyData.
%
% IN
% input_obj
% tolerance     allowed difference seen still as equal
%               (default: eps(single)
%
% OUT
% isObjectEqual true, if both objects are identical
% out_left - holds values of obj, which differ from input_obj
% out_right- holds values of input_obj, which differ from obj
%
% EXAMPLE
%   comp
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

oc = obj.copyobj;
ioc = input_obj.copyobj;
out_left = obj.copyobj;
out_right = input_obj.copyobj;
[out_right, isLeftObjectEqual] = out_right.diffobj(oc, tolerance);
[out_left, isRightObjectEqual] = out_left.diffobj(ioc, tolerance);

isObjectEqual = isLeftObjectEqual & isRightObjectEqual;
end