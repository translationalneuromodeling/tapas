function isEqual = uniqc_isempty(this)
% Tests whether an object is empty, i.e. whether it equals a newly created
% object from the same class.
%
%   Y = MrDimInfo()
%   Y.uniqc_isempty()
%
% This is a method of class MrDimInfo.
%
% IN
%
% OUT
%
% EXAMPLE
%   uniqc_isempty
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-08-17
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% get function handle of current class
className = class(this);
classHandle = str2func(className);
% create empty test object
testObj = classHandle();
% now comare
isEqual = isequal(this, testObj);
end