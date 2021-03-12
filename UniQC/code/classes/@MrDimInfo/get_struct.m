function [outputStruct, unusedArg] = get_struct(this, varargin)
% Returns the current dimInfo as a struct or creates a struct form the input.
%
%   Y = MrDimInfo()
%   Y.get_struct()
%   Y.get_struct('dimLabels', {'x', 'y', 'z'})
%
% This is a method of class MrDimInfo.
%
% IN
%
% OUT
%
% EXAMPLE
%   get_struct
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

% no input parameters given
if nargin == 1
    % transfrom this to struct
    warning('off', 'MATLAB:structOnObject');
    outputStruct = struct(this);
    warning('on', 'MATLAB:structOnObject');
    unusedArg = [];
    
else
    defaultDims = MrDimInfo();
    warning('off', 'MATLAB:structOnObject');
    defaultDims = struct(defaultDims);
    % set nDims empty as default
    defaultDims.nDims = [];
    warning('on', 'MATLAB:structOnObject');
    [outputStruct, unusedArg] = tapas_uniqc_propval(varargin(:), defaultDims);
end
end