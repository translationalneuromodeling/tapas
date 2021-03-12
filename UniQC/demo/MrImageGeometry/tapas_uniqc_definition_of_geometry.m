% Script definition_of_geometry
% Basic overview of how the image geomtry is represented in uniQC, in
% particular the combination of two affine matrices as defined by dimInfo
% and affineTrafo
%
% definition_of_geometry
%
%
% See also tapas_uniqc_spm_matrix tapas_uniqc_spm_imatrix MrImageGeometry

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-09-11
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1) Illustration of A = T * R * Z * S
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
resolution_mm   = [3 2 0.5];
nVoxels         = [30 110 180];
FOV_mm          = resolution_mm.*nVoxels;
offcentre_mm    = [110 80 -40];
shear           = [0 0.5 0];
rotation_deg    = [0 0 30];
% combine into P-vector as used by tapas_uniqc_spm_matrix
P               = [offcentre_mm rotation_deg/180*pi resolution_mm shear];

% translation
T   =   [1   0   0   P(1);
    0   1   0   P(2);
    0   0   1   P(3);
    0   0   0   1];
% rotation
R1  =   [1   0           0           0;
    0   cos(P(4))   sin(P(4))   0;
    0  -sin(P(4))   cos(P(4))   0;
    0   0           0           1];

R2  =   [cos(P(5))   0   sin(P(5))   0;
    0           1   0           0;
    -sin(P(5))  0   cos(P(5))   0;
    0           0   0           1];

R3  =   [cos(P(6))   sin(P(6))   0   0;
    -sin(P(6))  cos(P(6))   0   0;
    0           0           1   0;
    0           0           0   1];

R   = R1*R2*R3;

% zoom (scaling, resolution)
Z   =   [P(7)   0       0       0;
    0      P(8)    0       0;
    0      0       P(9)    0;
    0      0       0       1];

% shear
S   =   [1      P(10)   P(11)   0;
    0      1       P(12)   0;
    0      0       1       0;
    0      0       0       1];

% combine all to one affine transformation matrix A following the
% conventions in tapas_uniqc_spm_matrix
A = T*R*Z*S;

% origin of A is the voxel that is at [0 0 0]
% computed as the inverse of the transformation matrix
invA = inv(A);
origin = invA(1:3,4);
disp(origin);
% test by multiplying the origin voxel coordinates with the affine
% transformation - resulting vector is [0 0 0 1]
disp(A * [origin; 1]);

% ilustrate the effect of individual operations on two voxels
p1 = [1 1 1 1]';
p2 = [12 12 12 1]';

% shear only
disp(S*p1);
disp(S*p2);

% zoom only
disp(Z*p1);
disp(Z*p2);

% rotation only
disp(R*p1);
disp(R*p2);

% translation only
disp(T*p1);
disp(T*p2);

% translation and zoom
disp(T*Z*p1);
disp(T*Z*p2);

% affine transformation
disp(A*p1);
disp(A*p2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2) Within dimInfo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% only has resolution and translation
% in general, samplingPoints star counting at one when setting via
% resolutions and nSamples
% however, when loading from nifti, centre of matrix is centre of FOV

D = MrDimInfo('resolutions', resolution_mm, 'nSamples', nVoxels, ...
    'arrayIndex', [1 1 1], 'samplingPoint', -FOV_mm/2+resolution_mm/2);
% get affine matrix defined by dimInfo
DSP = [D.samplingPoints{'x'}(1), D.samplingPoints{'y'}(1), D.samplingPoints{'z'}(1)];
DR = D.resolutions;

DT  =   [1   0   0   DSP(1);
    0   1   0   DSP(2);
    0   0   1   DSP(3);
    0   0   0   1];

DZ   =  [DR(1)   0       0       0;
    0      DR(2)    0       0;
    0      0       DR(3)    0;
    0      0       0       1];

% affine transformation
AD = DT*DZ;
% equivalently: D.get_affine_matrix()

% origin of AD
invAD = inv(AD);
originD = [invAD(1:3,4); 1];
% equivalently: D.get_origin()
% centre of block is in origin
disp(AD*[nVoxels/2-0.5, 1]');
disp(AD * originD)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3) Combination
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% concatenate the affine transformation from affineTrafo and dimInfo
A_prime = A*AD;

% changing resolution in dimInfo
D2 = D.copyobj;
D2.resolutions = D.resolutions*2;

% computing new affine geoemtry
DSP2 = [D2.samplingPoints{'x'}(1), D2.samplingPoints{'y'}(1), D2.samplingPoints{'z'}(1)];
DR2 = D2.resolutions;

DT2  =   [1   0   0   DSP2(1);
          0   1   0   DSP2(2);
          0   0   1   DSP2(3);
          0   0   0   1];

DZ2   =  [DR2(1)   0       0       0;
          0      DR2(2)    0       0;
          0      0       DR2(3)    0;
          0      0       0       1];

% compute new combination
AD2 = DT2*DZ2;
A_Prime2 = A * AD2;

% check origin within dimInfo
originD2 = inv(AD2);
originD2 = [originD2(1:3, 4); 1];
disp(originD2);
disp(originD);
% --> origin is preserved

% now check origin of combined affineGeom
origin_A_prime = inv(A_prime);
origin_A_prime2 = inv(A_Prime2);
disp(origin_A_prime(1:3, 4));
disp(origin_A_prime2(1:3, 4));
% --> origin is move due to the second affine geometry

% change resolution while preserving origin
A_preserved_origin = A_prime/AD2;
A_prime2_preserved_origin = A_preserved_origin * AD2;
% -->  this is equal to A_prime; however, A has to be changed (in
% MrAffineTransformation)