function this = set_from_dimInfo_and_affineTrafo(this, dimInfo, affineTransformation)
% Creates MrImageGeometry from MrDimInfo and MrAffineTransformation
%
%   Y = MrImageGeometry()
%   Y.set_from_dimInfo_and_affineTrafo(dimInfo, affineTransformation)
%
% This is a method of class MrImageGeometry.
%
% IN
%
% OUT
%
% EXAMPLE
%   dimInfo = MrDimInfo(fileName);
%   affineTransformation = MrAffineTransformation(fileName);
%   ImageGeometry = MrImageGeometry(dimInfo, affineTransformation);
%
%   See also MrImageGeometry

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-10-30
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% check input
isValidInput = (isa(dimInfo, 'MrDimInfo')) && (isa(affineTransformation, 'MrAffineTransformation'));

if isValidInput
    % Concatenate affine geometries as defined by dimInfo and affineTrafo.
    
    % The definition of an affine matrix (A) follows SPM and is given by
    % A = T*R*Z*S, with T = translation , R = rotation, Z = zoom (scaling)
    % and S = shear. See tapas_uniqc_spm_matrix for more details.
    
    % The final affine image geometry is computed from
    % AImage = AAffineTrafo * ADimInfo
    % (TImage*RImage*ZImage*SImage) =
    % (TAffineTrafo*RAffineTrafo*SAffineTrafo) * (TDimInfo*ZDimInfo).
    
    AAffineTrafo = affineTransformation.affineMatrix;
    ADimInfo = dimInfo.get_affine_matrix;
    % compute combined affine matrix
    AImage = AAffineTrafo * ADimInfo;
    
    % split into individual affine operations (shift, rot etc.)
    % but round to significant decimals of double precision
    N = floor(abs(log10(eps('double'))));
    P = round(tapas_uniqc_spm_imatrix(AImage),N);
    
    % populate fields with affine transformations
    this.offcenter_mm       = P(1:3);
    this.rotation_deg       = P(4:6)/pi*180;
    this.resolution_mm      = P(7:9);
    this.shear              = P(10:12);
    
    % properties from MrDimInfo
    this.nVoxels(1:3) = 1;
    % x
    if ~isempty(dimInfo.nSamples('x'))
        this.nVoxels(1) = dimInfo.nSamples('x');
    end
    % y
    if ~isempty(dimInfo.nSamples('y'))
        this.nVoxels(2) = dimInfo.nSamples('y');
    end
    % z
    if ~isempty(dimInfo.nSamples('z'))
        this.nVoxels(3) = dimInfo.nSamples('z');
    end
    
    % search for timing info
    trCharacters = {'t', 'time', 'TR'};
    trFound = ismember(trCharacters, dimInfo.dimLabels);
    if any(trFound)
        % check whether timing information is given in (milli)seconds
        if strcmpi('s', dimInfo.units(trCharacters{trFound}))
            this.TR_s = dimInfo.resolutions(trCharacters{trFound});
            this.nVoxels(4) = dimInfo.nSamples(trCharacters{trFound});
        elseif strcmpi('ms', dimInfo.units(trCharacters{trFound}))
            this.TR_s = dimInfo.resolutions(trCharacters{trFound})./1000;
            this.nVoxels(4) = dimInfo.nSamples(trCharacters{trFound});
        else
            this.nVoxels(4) = dimInfo.nSamples(trCharacters{trFound});
        end
    end
    
    % compute FOV directly; ignoring that there might only be nVoxels-1
    % gaps
    this.FOV_mm = this.nVoxels(1:3).*this.resolution_mm;
else
    fprintf('Geometry could not be created: Invalid Input (MrDimInfo and MrAffineTransformation expected');
end