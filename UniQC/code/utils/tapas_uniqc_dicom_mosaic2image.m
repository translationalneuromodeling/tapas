function X = tapas_uniqc_dicom_mosaic2image(pathDicom)
% Computes MrImage from Dicom-Folder with mosaic images (over slices), one
% image per volume
%
%   X = tapas_uniqc_dicom_mosaic2image(pathDicom)
%
% IN
%   pathDicom   folder with mosaic-style *.IMA files
% OUT
%
% EXAMPLE
%   tapas_uniqc_dicom_mosaic2image
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-02-02
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%

if nargin < 1
    pathDicom = pwd;
end

%%
d = dir(fullfile(pathDicom, '*.IMA'));
fileNames = strcat(pathDicom, filesep, {d.name}');

nFiles = numel(fileNames);

%% read first dicom file header to determine dimensions

info = dicominfo(fileNames{1});
resolutions = zeros(1,4);
resolutions(1:2) = double(info.PixelSpacing);
resolutions(4) = info.RepetitionTime/1000;
resolutions(3) = info.SpacingBetweenSlices;
samplingWidths = resolutions;
samplingWidths(3) = info.SliceThickness;

% The following metadata are probably site/release-specific
nSamples = zeros(1,4);
nSamples(1:2) = double(sscanf(info.Private_0051_100b, '%d*%d'));

% TODO: rather use NumberOfImagesInMosaic from private Siemens header
% (CSA)
% info=spm_dicom_headers(file)
% info{1}.CSAImageHeaderInfo(22).item(1)
% ...or is that the same?
nSamples(3) = info.Private_0019_100a;
nSamples(4) = nFiles;

% TODO: check whether 1&2 should be reversed
nSlicesMosaicRow = info.Rows/nSamples(1);
nSlicesMosaicCol = info.Columns/nSamples(2);

data = zeros(nSamples);

%% read all files in loop
for n = 1:nFiles
    mosaicData = dicomread(fileNames{n});
    volData = reshape(permute(reshape(mosaicData, ...
        [nSamples(1) nSlicesMosaicCol nSamples(2) nSlicesMosaicRow]), ...
        [1 3 4 2]), ...
        nSamples(1), nSamples(2), []);
    data(:,:,:,n) = volData(:,:,1:nSamples(3));
end

%%

FOV = (nSamples.*resolutions);
dimInfo = MrDimInfo('resolutions', resolutions, 'nSamples', nSamples, ...
    'firstSamplingPoint', [-FOV(1:3)/2, 0], 'samplingWidths', samplingWidths);

X = MrImage(data, 'dimInfo', dimInfo);

% create reasonable name for image
nameImage = pathDicom;
nameImage(1:end-30) = [];
nameImage = ['...' nameImage];
X.name = nameImage;

%% TODO: adjust affine geometry according to http://nipy.org/nibabel/dicom/dicom_mosaic.html