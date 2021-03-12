function outputImage = rotate(this, rot_deg)
% Geometric rotation of MrImage
%
%   Y = MrImage()
%   Y.rotate(rot_deg)
%
% This is a method of class MrImage.
%
% NOTE: This is a method of MrImage rather than MrImageGeometry, because
%       the latter is composed on the fly from affineTransformation and
%       dimInfo to integrate both information and sustain consistency.
%       Thus, the effect will only be visible using the world space plot
%       options, i.e. Y.plot('plotType', 'spmi').
% NOTE: The rotation is *added* to the rotation already defined in the
%       affine transformation matrix, equivalent to the rotation in SPM
%       Display.
% IN
%   rot_deg             [1,3] rotation delta components, i.e., [rx, ry, rz]
%                       in degree
% OUT
%
% EXAMPLE
%   rotate
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-12-03
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

outputImage = this.copyobj();

[outputImage.dimInfo, outputImage.affineTransformation] = ...
    this.geometry.perform_world_space_operation('rotation', rot_deg, this.dimInfo);

end