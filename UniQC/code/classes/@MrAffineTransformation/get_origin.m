function origin = get_origin(this)
% Computes the origin of the affine transformation.
%
%   Y = MrAffineTransformation()
%   Y.get_origin()
%
% This is a method of class MrAffineTransformation.
%
% IN
%
% OUT   Voxel indices x, y, z of the origin, i.e. those voxel who are
%       at location [0 0 0].
%
% EXAMPLE
%   get_origin
%
%   See also MrImageGeometry demo/MrImageGeometry/definition_of_geometry
 
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

invA = inv(this.get_affine_matrix());
origin = invA(1:3,4);
end