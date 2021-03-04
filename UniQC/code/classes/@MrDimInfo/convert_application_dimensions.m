function applicationDimensions = convert_application_dimensions(this, ...
    applicationDimensions)
% converts chars of applicationDimensions to indices, and returns error, if
% 
% dimension index as default, if char is not a valid label for any dimension
%
%   Y = MrDimInfo()
%   applicationDimensions = convert_application_dimensions(Y, ...
%    applicationDimensions)
%
% This is a method of class MrDimInfo.
%
% IN
%   applicationDimensions   double vector or char
%                           if char, dim labels will be converted to dim index
%                           special keywords
%                           '2d' defaults to [1,2]
%                           '3d' defaults to [1,2,3]
%                           non-existent label defaults to error
% OUT
%
% EXAMPLE
%   convert_application_dimensions
%
%   See also MrDimInfo
 
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-07-23
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if ischar(applicationDimensions)
    switch lower(applicationDimensions)
        case '2d'
            applicationDimensions = [1 2];
        case '3d'
            applicationDimensions = [1 2 3];
        otherwise % use dimInfo to determine dimension
            applicationDimensions = this.get_dim_index(applicationDimensions);
            % default: last dimension, if nothing found
            if isempty(applicationDimensions)
               error('tapas:uniqc:MrDimInfo:InvalidDimLabel', ...
                   'DimLabel %s is not a valid dim label of this dimInfo', ...
                   applicationDimensions);
            end
    end
end