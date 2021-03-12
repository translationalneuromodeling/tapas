% Script demo_set_geometry
% Illustrates the different components that define an image geometry.
%
%  demo_set_geometry
%
%
%   See also definition_of_geometry
%
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-11-05
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
%% 1. Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get example data
dataPath = tapas_uniqc_get_path('data');
niftiFile4D = fullfile(dataPath, 'nifti', 'rest', 'fmri_short.nii');
dataLoad = MrImage(niftiFile4D);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Create MrImage object from matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% no geometry information is supplied, defaults are used
dataRaw = MrImage(dataLoad.data); % as reference, remains unchanged
data = MrImage(dataLoad.data); % geometry and dimInfo will be adapted
dataRaw = dataRaw.select('t', 1);
data = data.select('t', 1);
data.parameters.save.fileName = 'orig.nii';
disp_centre_and_origin(data);
data.plot('plotType', 'spmi');
%   Note 1: no dimInfo/geometry information are supplied, defaults are used:
%   nSamples is derived from the data matrix
%   resolutions is assumed to be 1
%   the origin (voxel at position [0 0 0] mm) is in the center of the image

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Add resolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot data using the classical way, but adding sampling points
% (this option is only available with the image processing toolbox)
iptsetpref('ImshowAxesVisible', 'on');
f = data.plot;
a = f.CurrentAxes;
nX = round(a.XLim(2)/data.dimInfo.nSamples(1));
xAxis = repmat(data.dimInfo.samplingPoints{1}, [1,nX]);
a.XTickLabel = xAxis(a.XTick);
nY = round(a.YLim(2)/data.dimInfo.nSamples(2));
yAxis = repmat(data.dimInfo.samplingPoints{2}, [1,nY]);
a.YTickLabel = yAxis(a.YTick);

% add additional resolution information
data.dimInfo.resolutions = dataLoad.dimInfo.resolutions;

f = data.plot;
a = f.CurrentAxes;
nX = round(a.XLim(2)/data.dimInfo.nSamples(1));
xAxis = repmat(data.dimInfo.samplingPoints{1}, [1,nX]);
a.XTickLabel = xAxis(a.XTick);
nY = round(a.YLim(2)/data.dimInfo.nSamples(2));
yAxis = repmat(data.dimInfo.samplingPoints{2}, [1,nY]);
a.YTickLabel = yAxis(a.YTick);

disp_centre_and_origin(data);
data.plot('plotType', 'spmi', 'overlayImages', dataRaw);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Add Shear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% none of these options will affect matrix plot
data.affineTransformation.shear = [0.5 0 0];
data.plot('plotType', 'spmi', 'overlayImages', dataRaw);
disp_centre_and_origin(data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Add Rotation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data.affineTransformation.shear = [0 0 0];
data.affineTransformation.rotation_deg = [0 30 0];
data.plot('plotType', 'spmi', 'overlayImages', dataRaw);
disp_centre_and_origin(data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4a. Add Translation (offcentre_mm) in the affineTrafo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data.affineTransformation.offcenter_mm(3) = 10;
data.plot('plotType', 'spmi', 'overlayImages', dataRaw);
% world space origin is changed (but note, that, since the transformation is
% applied last, the origin changes in the two dimension which are affected by the rotation) 
disp_centre_and_origin(data);
% but voxel space origin is maintained
disp(data.dimInfo.get_origin());

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4b. Change translation in dimInfo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% now the translation is applied first
data2 = data.select('t', 1).copyobj();
data2.affineTransformation.offcenter_mm(3) = -10;
data2.dimInfo.set_dims(3, 'firstSamplingPoint', data2.dimInfo.samplingPoints{3}(1) + 20);
data2.plot('plotType', 'spmi', 'overlayImages', data.select('t', 1));
disp_centre_and_origin(data2);
