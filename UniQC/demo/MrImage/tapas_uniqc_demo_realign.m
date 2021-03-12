% Script demo_realign
% Shows realignment for n-dimensional data with different scenarios of 4D
% subsets feeding into estimation, and parameters applied to other subsets,
% e.g.
%   - standard 4D MrImageSpm realignment, with or without weighting of
%     particular voxels
%   - multi-echo data, 1st echo realigned, applied to all echoes
%   - complex data, magnitude data realigned, phase data also shifted
%   - multi-coil data, root sum of squares realigned, applied to each coil
%
%  demo_realign
%
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-25
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
%% 1a. 4D fMRI, real valued, standard realignment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathExamples = tapas_uniqc_get_path('examples');
fileTest = fullfile(pathExamples, 'nifti', 'rest', 'fmri_short.nii');

Y = MrImage(fileTest);
[rY,rp] = Y.realign();

plot(Y-rY, 't', 1:Y.dimInfo.nSamples('t'), 'z', 23, 'sliceDimension', 't');
plot(rY.snr - Y.snr, 'displayRange', [-5 5], 'colorMap', 'hot');

figure('WindowStyle', 'docked');
plot(rp(:,1:3), '-'); hold all; plot(rp(:,4:6), '--');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1b. 4D fMRI, real valued, weighted realignment with manual mask
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mask including only 90 percentile mean voxel intensities
M = Y.mean('t');
M = M.threshold(M.prctile(90));

[rYM,rpM] = Y.realign('weighting', M);

plot(Y-rYM, 't', 1:Y.dimInfo.nSamples('t'), 'z', 23, 'sliceDimension', 't');
plot(rYM.snr - Y.snr, 'displayRange', [-5 5], 'colorMap', 'hot');

figure('WindowStyle', 'docked');
plot(rpM(:,1:3), '-'); hold all; plot(rpM(:,4:6), '--');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. 5D multi-echo fMRI, realignment variants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathExamples = tapas_uniqc_get_path('examples');
pathMultiEcho = fullfile(pathExamples, 'nifti', 'data_multi_echo');

% loads all 4D nifti files (one per echo) in 5D array; takes dim name of
% 5th dimension from file name
ME = MrImage(fullfile(pathMultiEcho, 'multi_echo*.nii'));
ME.dimInfo.set_dims(5, 'dimLabels', 'echo');

%% Realign 10 volumes via 1st echo
% the default is to use the first 4 dimensions ans apply the estimated
% parameters along all higher dimensions
% e.g. the first echo is used to estimate the realignment parameters and
% these are then applied to all three echoes
rME = ME.realign;
plot(rME-ME, 't', 11);

%% Realign 10 volumes via mean of echoes
meanI = ME.mean('echo');
r2ME = ME.realign('representationIndexArray', meanI, ...
    'applicationIndexArray', {'echo', 1:3});
plot(r2ME-ME, 't', 11);
plot(r2ME.mean('echo').snr('t') - ME.mean('echo').snr('t'), 'displayRange', [-10 10]);
ME.plot('echo', 1, 'z', 23, 'sliceDimension', 't');

% output plots to check whether all echoes were indeed realigned
ME.plot('echo', 1, 'z', 23, 't', 1);
ME.plot('echo', 1, 'z', 23, 't', 11);
ME.plot('echo', 1, 'z', 23, 't', 1);
r2ME.plot('echo', 1, 'z', 23, 't', 11);

ME.plot('echo', 3, 'z', 23, 't', 1);
r2ME.plot('echo', 3, 'z', 23, 't', 11);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. 4D fMRI, complex valued, realignment of magnitude and also applied to phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nSamples = [48, 48, 9, 3];
data = randn(nSamples);
dataReal = tapas_uniqc_create_image_with_index_imprint(data);
% to change orientation of imprint in imag part
dataImag = permute(tapas_uniqc_create_image_with_index_imprint(data),[2 1 3 4]);
IComplex = MrImage(dataReal+1i*dataImag, ...
    'dimLabels', {'x', 'y', 'z', 't'}, ...
    'units', {'mm', 'mm', 'mm', 's'}, ...
    'resolutions', [1.5 1.5 3 2], 'nSamples', nSamples);

IComplex.real.plot();
IComplex.imag.plot();
rIComplex = IComplex.realign;
plot(IComplex.abs - rIComplex.abs, 't', IComplex.dimInfo.t.nSamples);
rIComplexMP = IComplex.realign('splitComplex', 'ri');
plot(rIComplex.abs - rIComplexMP.abs, 't', IComplex.dimInfo.t.nSamples);
