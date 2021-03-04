% Script demo_change_geometry
% How to change the gemetry parameters (shift, rot, zoom, shear) in
% MrImageGeometry or MrDimInfo (axis-parallel shift and zoom only)
%
%  demo_change_geometry
%
%
%   See also

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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0. Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
doInteractivePlot = 1;
% get example data
dataPath = tapas_uniqc_get_path('data');
niftiFileStruct = fullfile(dataPath, 'nifti', 'rest', 'struct.nii');
mLoad = MrImage(niftiFileStruct);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Make up geometry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = MrImage(mLoad.data);
% only select a few slices for visualisation
m.plot('z', 1:10:m.dimInfo.nSamples('z'));
if doInteractivePlot, m.plot('plotType', 'spmi'); end
% set start geometry
m.dimInfo.resolutions = [1 1.5 2];
m.plot('z', 1:10:m.dimInfo.nSamples('z'));
% set start offcenter
m.dimInfo.set_dims(3, 'firstSamplingPoint', m.dimInfo.samplingPoints{3}(1)+10);
m.plot('z', 1:10:m.dimInfo.nSamples('z'));

% all over parameters are set in affineTransformation and will not impact
% the plot
m.affineTransformation.rotation_deg = [10 0 0];
m.affineTransformation.shear = [0 0.05 0];
disp(m.affineTransformation);
if doInteractivePlot, m.plot('plotType', 'spmi'); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Changes in world space (just as SPM display would do)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this is our starting geometry
disp(m.geometry);
% we cannot change m.geometry directly, because it is always derived on the
% fly from affineTransformation and dimInfo
% changes in world space do not affect dimInfo

shiftedImage = m.shift([0 100 0]);
disp(shiftedImage.geometry);
if doInteractivePlot, m.plot('plotType', 'spmi', 'overlayImages', shiftedImage); end

rotatedImage = m.rotate([0 0 45]);
disp(rotatedImage.geometry);
if doInteractivePlot, m.plot('plotType', 'spmi', 'overlayImages', rotatedImage); end

shearedImage = m.shear([0.5 0 0]);
disp(shearedImage.geometry);
if doInteractivePlot, m.plot('plotType', 'spmi', 'overlayImages', shearedImage); end

zoomedImage = m.zoom([2 4 3]);
disp(zoomedImage.geometry);
if doInteractivePlot, m.plot('plotType', 'spmi', 'overlayImages', zoomedImage); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Setting to a specific value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% setting to a specific value
% rotation_mm
mRot = m.set_geometry_property('rotation_deg', [0 0 30]);
if doInteractivePlot, m.plot('plotType', 'spmi', 'overlayImages', mRot); end
disp(mRot.geometry);

% offcenter_mm
mOff = m.set_geometry_property('offcenter_mm', [0 0 0]);
if doInteractivePlot, m.plot('plotType', 'spmi', 'overlayImages', mOff); end
disp(mOff.geometry);

% shear
mShear = m.set_geometry_property('shear', [0 0.2 0]);
if doInteractivePlot, m.plot('plotType', 'spmi', 'overlayImages', mShear); end
disp(mShear.geometry);

% resolution_mm
mRes = m.set_geometry_property('resolution_mm', [1 2 1]);
if doInteractivePlot, m.plot('plotType', 'spmi', 'overlayImages', mRes); end
disp(mRes.geometry);


% now changing the origin, i.e. axis parallel shifts
setO = m.copyobj();
affineM = setO.geometry.get_affine_matrix();
invAffineM = inv(affineM);
% now set origin
invAffineM(1:3, 4) = [60 50 10];
newAffineM = inv(invAffineM);

newAffineT = newAffineM/setO.dimInfo.get_affine_matrix();
setO.affineTransformation.update_from_affine_matrix(newAffineT);

disp(m.geometry);
disp(setO.geometry);

disp(m.geometry.get_origin());
disp(setO.geometry.get_origin());