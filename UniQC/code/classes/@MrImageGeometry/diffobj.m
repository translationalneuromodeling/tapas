function [diffGeometry, isEqual, isEqualGeom3D]  = diffobj(this, targetGeometry, tol)
% Compares image geometries, and gives detailed info about differences
% (2D/3D/4D, other params)
%
%   Y = MrImageGeometry()
%   Y.diffobj(inputs)
%
% This is a method of class MrImageGeometry.
%
% IN
%   inputGeom   MrImageGeometry to compare
%
% OUT
%   diffGeom    MrImageGeometry with properties set that are different in
%               this and inputGeom
%   iEqual      true, if all properties are identical
%   isEqualGeom3D true, if all
%
% EXAMPLE
%   diffobj
%
%   See also MrImageGeometry MrCopyData.diffobj

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-11-15
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 3
    tol = eps('single');
end

[diffGeometry, isEqual] = diffobj@MrCopyData(this, targetGeometry, tol);

isEqualGeom3D = isEqual;

% if only 4th geometry dimension is unequal, or only TR, between, no problem, no resize
% necessary!
if ~isEqualGeom3D
    fieldNamesDiff = diffGeometry.get_nonempty_fields;
    
    % accept differences in nVolumes and TR as still same geometry
    isEqualGeom3D = all(ismember(fieldNamesDiff, {'nVoxels', 'TR_s'}));
    
    % Check voxel size precisely
    if isEqualGeom3D
        
        isEqualGeom3D = ...
            isequal(this.nVoxels(1:3), targetGeometry.nVoxels(1:3));
    end
end
