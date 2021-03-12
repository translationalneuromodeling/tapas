% Script demo_smooth
% Shows smoothing
%
%  demo_smooth
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-27
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


clear;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. 4D fMRI, real valued, standard smoothing via SPM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathExamples = tapas_uniqc_get_path('examples');
fileTest = fullfile(pathExamples, 'nifti', 'rest', 'fmri_short.nii');

% load data
Y = MrImage(fileTest);

sY = Y.smooth();                    % default smoothing, 8 mm fwhm
s3Y = Y.smooth('fwhm', 3);          % non-default smoothing
s381Y = Y.smooth('fwhm', [3 8 1]);  % anisotropic smoothing

% check how smoothly it went...
Y.plot();
sY.plot();
s3Y.plot();
s381Y.plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. 4D fMRI, complex valued, smoothing of separate parts (real/imag vs magn/phase)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nSamples = [48, 48, 9, 3];
data = randn(nSamples);
dataReal = tapas_uniqc_create_image_with_index_imprint(data);
% to change orientation of imprint in imag part
dataImag = permute(tapas_uniqc_create_image_with_index_imprint(data),[2 1 3 4]);
YComplex = MrImage(dataReal+1i*dataImag, ...
    'dimLabels', {'x', 'y', 'z', 't'}, ...
    'units', {'mm', 'mm', 'mm', 's'}, ...
    'resolutions', [1.5 1.5 3 2], 'nSamples', nSamples);

YComplex.real.plot();
YComplex.imag.plot();

%plot real and imaginary part separately (Default)
sriYComplex = YComplex.smooth('fwhm',3);
sriYComplex.real.plot();
sriYComplex.imag.plot();

smpYComplex = YComplex.smooth('fwhm',3,'splitComplex','mp');
smpYComplex.real.plot();
smpYComplex.imag.plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. 5D multi-echo fMRI, smoothing variants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathExamples = tapas_uniqc_get_path('examples');
pathMultiEcho = fullfile(pathExamples, 'nifti', 'multi_echo_data');

ME = MrImage(pathMultiEcho);

% smooth every echo separately as 4D
sME = ME.smooth();

ME.plot('t', 1);
sME.plot('t', 1);