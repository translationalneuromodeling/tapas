% Script demo_image_operations
% Examples of image operation methods of MrImage.
%
%  demo_image_operations
%
%
%   See also
 
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-11-22
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
 
clear;
close all;
clc; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0. Load example data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathExamples    = tapas_uniqc_get_path('examples');
fileTest        = fullfile(pathExamples, 'nifti', 'rest', 'meanfmri.nii');
X               = MrImage(fileTest);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Flip
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X.plot();
%  flip up-down (first dim), left-right (second dim), and in z
X.flipud.plot();
X.fliplr.plot();
X.flip('z').plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Threshold image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X.threshold(600).plot();
X.threshold([600, 850]).plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Remove slice
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% remove first 5 and last slice
s = X.select('z', [1:5, X.dimInfo.nSamples('z')], 'invert', true);
X.plot('plotType', 'spmi', 'overlayImages', s);
s2 = X.select('z', 1:2:X.dimInfo.nSamples('z'));
X.plot('plotType', 'spmi', 'overlayImages', s2);