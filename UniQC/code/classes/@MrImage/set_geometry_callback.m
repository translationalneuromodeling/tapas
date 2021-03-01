function set_geometry_callback(this, obj, eventData)
% for set-functionality of properties of MrImageGeometry within MrImage
%
%   output = set_geometry_callback(input)
%
% IN
%
% OUT
%
% EXAMPLE
%   set_geometry_callback
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-11-02
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

warning('Set-Method for geometry does not exist. Change dimInfo or affineTransformation');