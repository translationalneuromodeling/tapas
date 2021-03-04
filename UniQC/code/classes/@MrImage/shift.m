function outputImage = shift(this, dr, worldOrVolumeSpace)
% geometric shift of MrImage
%
%   Y = MrImage()
%    outputImage = Y.shift(dr, worldOrVolumeSpace)
%
% This is a method of class MrImage.
%
% NOTE: This is a method of MrImage rather than MrImageGeometry, because
%       the latter is composed on the fly from affineTransformation and
%       dimInfo to integrate both information and sustain consistency.
%
% NOTE: The shift is *added* to the translation already defined in the
%       affine transformation matrix, equivalent to the shift in SPM
%       Display.
%
% IN
%   dr                  [1,3] translation delta components, i.e., [dx, dy, dz]
%   worldOrVolumeSpace  'world' or 'volume'
%                       defines whether transformation shall be performed
%                       in world space
%                           - relative to origin of MrImageGeometry
%                           - dimInfo remains unaltered
%                           - affineTransformation changes
%                       in volume space (i.e. respecting resolutions and
%                       axis-parallel shifts of origin, e.g., to define
%                       center of a scanner acquisition volume)
%                           - relative to origin of dimInfo-defined volume
%                           - dimInfo changes
%                           - affineTransformation changes
%                           - overall, MrImageGeometry stays constant
%
% OUT
%   outputImage         new image with beautiful new geometry
%
% EXAMPLE
%   shift
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-12-02
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 3
    worldOrVolumeSpace = 'world';
end

outputImage = this.copyobj();

switch worldOrVolumeSpace
    case 'world'
        [outputImage.dimInfo, outputImage.affineTransformation] = ...
            this.geometry.perform_world_space_operation('shift', dr, this.dimInfo);
    case 'volume'
    otherwise
        error('tapas:uniqc:MrImage:shift:UnknownSpace', ...
            '%s...A space where no one has gone before', worldOrVolumeSpace);
end