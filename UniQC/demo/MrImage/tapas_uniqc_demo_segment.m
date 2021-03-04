% Script demo_segment
% Illustrates usage of segment for 3D and 4D images, and the additional
% parameters available in the batch editor
%
%  demo_segment
%
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-12-01
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
%% 0. Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pathData            = tapas_uniqc_get_path('examples');
fileFunctionalMean  = fullfile(pathData, 'nifti', 'rest', 'struct.nii');
m = MrImage(fileFunctionalMean);

plotString = {'z', 1:10:m.dimInfo.nSamples('z'), 'rotate90', -1};

m.plot(plotString{:});
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Segment image with additional outputs and SPM parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the outputs are the tissue probability maps (can be used for maks
% generation, for example), the deformation fields, the (inverse) bias
% field and the bias corrected image
% Note: sampling distance is increased from its default value (3) to speed
% up the segmentation process

%% A) all output parameters
[biasFieldCorrected, tissueProbMaps, deformationFields, biasField] = ...
    m.segment('samplingDistance', 20, 'deformationFieldDirection', 'both');

biasFieldCorrected.plot(plotString{:});
nTPM = numel(tissueProbMaps);
for n = 1:nTPM
    tissueProbMaps{n}.plot(plotString{:});
end
deformationFields{1}.plot();
deformationFields{2}.plot(plotString{:});
biasField{1}.plot(plotString{:});

%% B) all tissue types, larger bias FWHM, no clean up
tissueTypes = {'WM', 'GM', 'CSF', 'bone', 'fat', 'air'};
biasRegularisation = 1e-4;
biasFWHM = 90;
cleanUp = 0;

[biasFieldCorrected2, tissueProbMaps2, deformationFields2, biasField2] = ...
    m.segment('samplingDistance', 20, 'tissueTypes', tissueTypes, ...
    'biasRegularisation', biasRegularisation, 'biasFWHM', biasFWHM, ...
    'cleanUp', 0);

biasFieldCorrected2.plot(plotString{:});
nTPM2 = numel(tissueProbMaps2);
for n = 1:nTPM2
    tissueProbMaps2{n}.plot(plotString{:});
end
deformationFields2{1}.plot();
biasField2{1}.plot(plotString{:});

%% C) output maps in mni space
[biasFieldCorrected, tissueProbMapsMni, deformationFieldsMni, biasField] = ...
    m.segment('samplingDistance', 20, 'mapOutputSpace', 'warped');
biasFieldCorrected.plot();
nTPMMni = numel(tissueProbMapsMni);
for n = 1:nTPMMni
    tissueProbMapsMni{n}.plot;
end
deformationFieldsMni{1}.plot();
biasField{1}.plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Segment 5D image with additional contrasts (channels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load multi-echo data
pathData = tapas_uniqc_get_path('examples');
pathMultiEcho = fullfile(pathData, 'nifti', 'data_multi_echo');
ME = MrImage(fullfile(pathMultiEcho, 'multi_echo*.nii'));
TE = [9.9, 27.67 45.44];
ME.dimInfo.set_dims('echo', 'units', 'ms', 'samplingPoints', TE);
ME.dimInfo.set_dims('t', 'resolutions', 0.65);

% this is a toy example, so we only choose a few time points
MESmall = ME.select('t', [1,3], 'echo', [1,2]);

% segment
% note that all dimensions except x, y and z will be treated as additional
% channels
[biasFieldCorrectedMc, tissueProbMapsMc, deformationFieldsMc, biasFieldMc] = ...
    MESmall.segment('samplingDistance', 10);
for t = 1:MESmall.dimInfo.t.nSamples
    MESmall.plot('z', 23, 't', t, 'sliceDimension', 'echo', 'displayRange', [0 1400]);
    biasFieldCorrectedMc.plot('z', 23, 't', t, 'sliceDimension', 'echo', 'displayRange', [0 1400]);
    biasFieldMc{1}.plot('z', 23, 't', t, 'sliceDimension', 'echo');
end

nTPMMc = numel(tissueProbMapsMc);
for n = 1:nTPMMc
    tissueProbMapsMc{n}.plot;
end
for n = 1:nTPMMc
    MESmall.mean('echo').plot('z', 23, 't', 1);
    tissueProbMapsMc{n}.plot('z', 23, 'displayRange', [0 1]);
end
deformationFieldsMc{1}.plot;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Segment complex image (split into magnitude/phase is the default)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% make complex image
cm = m.copyobj();
% just add noise for the imaginary part
cmI = (cm + 300*randn(cm.dimInfo.nSamples)).*(1i);
cm.data = cm.data + cmI.data;

% plot real and imaginary part
cm.real.plot();
cm.imag.plot();
% segment
bcm = cm.segment();
% plot bias fiel corrected real and imaginary part
bcm.real.plot();
bcm.imag.plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Segment each echo individually
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute mean over time
MEmean = ME.mean('t');
% segment each echo
[MEmean_B, MEmean_TPM, MEmean_DF, MEmean_BF] = MEmean.segment('representationIndexArray', ...
    {{'echo', 1}, {'echo', 2}, {'echo', 3}}, 'samplingDistance', 10);
% plot results
MEmean_B.plot('z', 30, 'sliceDimension', 'echo', 'displayRange', [0 1400]);
for n = 1:numel(MEmean_TPM)
    MEmean_TPM{n}.plot('z', 30, 'sliceDimension', 'echo', 'displayRange', [0 1]);
end