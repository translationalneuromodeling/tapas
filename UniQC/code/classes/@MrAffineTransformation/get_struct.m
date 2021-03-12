function [outputStruct, unusedArg] = get_struct(this, varargin)
% Returns the current affineTransformation as a struct or creates struct
% from input.
%
%   Y = MrAffineTransformation()
%   YasStruct = Y.get_struct()
%   Y.get_struct('shear', [0.5 0.1 -0.3])
%
% This is a method of class MrAffineTransformation.
%
% IN
%       MrAffineTransformation object (this)
%       Property value pairs for properties of MrAffineTransformation
%
% OUT
%       Struct that contains values only at places where information was
%       available (i.e. defaults are set to empty) and unused Arguments.
%
% EXAMPLE
%   get_struct
%
%   See also MrAffineTransformation

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-12-04
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin == 1
    % transfrom this to struct
    warning('off', 'MATLAB:structOnObject');
    outputStruct = struct(this);
    warning('on', 'MATLAB:structOnObject');
    unusedArg = [];
    
else
    defaultAffineTrafo = MrAffineTransformation();
    warning('off', 'MATLAB:structOnObject');
    defaultAffineTrafo = struct(defaultAffineTrafo);
    % set empty as default
    for thisField = fieldnames(defaultAffineTrafo)'
        defaultAffineTrafo.(thisField{1}) = [];
    end
    warning('on', 'MATLAB:structOnObject');
    [outputStruct, unusedArg] = tapas_uniqc_propval(varargin(:), defaultAffineTrafo);
end
end
