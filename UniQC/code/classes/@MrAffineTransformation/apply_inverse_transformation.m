function this = apply_inverse_transformation(this, otherGeometry)
% Applies inverse of given transformation (as an Image geometry) to this
%
%(e.g. given mapping stationary -> transformed image transformation that
% now shall be applied to transformable image to warp into space of
% stationary image )
%
%   Y = MrAffineTransformation()
%   Y.apply_inverse_transformation(otherGeometry)
%
% This is a method of class MrAffineTransformation.
%
% IN
%   otherGeometry   MrAffineTransformation holding the affine transformation to be
%                   applied
%
% OUT
%
% EXAMPLE
%   apply_inverse_transformation
%
%   See also MrAffineTransformation

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-28
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% in spm_coreg: MM
rawAffineMatrix = this.affineMatrix;

% in spm_coreg: M
if ~isa(otherGeometry, 'MrAffineTransformation')
    % Input parameter not an MrAffineTransformation, assuming affine Matrix
    otherGeometry = MrAffineTransformation(otherGeometry);
end
affineCoregistrationMatrix = otherGeometry.affineMatrix;

% compute inverse transformation via \, efficient version of:
% pinv(affineCoregistrationMatrix) * rawAffineMatrix 
processedAffineMatrix = affineCoregistrationMatrix \ ...
    rawAffineMatrix;
this.update_from_affine_matrix(processedAffineMatrix);