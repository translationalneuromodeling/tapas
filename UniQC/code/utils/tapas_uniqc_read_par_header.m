function header = tapas_uniqc_read_par_header(filename)
% Returns Par-file header (Philips), based on GyroTools ReadRecV3
%
%  header = tapas_uniqc_read_par_header(fileName)
%
% IN
%   fileName    .par/.rec file
%
% OUT
%
% EXAMPLE
%   tapas_uniqc_read_par_header
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper & Laetitia Vionnet
% Created:  2016-01-31
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


[fp, fn, ext] = fileparts(filename);

parFile = fullfile(fp, [fn '.par']);

% load voxel dimensions from par-file
% read voxel size from pixel spacing and thickness+gap
%#sl   ec  dyn ph  ty   idx    pix %   rec size       (re)scale                   window       angulation_deg        offcenter_mm             thick  gap     info     spacing      echo  dtime ttime diff   avg flip    freq  RR_int turbo delay b grad cont      anis              diffusion      L.ty
% 1    1   1   1   0  2 0      16 100  224  224        0  12.2877 5.94433e-003   1070   1860 -0.75  0.11  0.67  -1.644 -12.978  -1.270 0.80   0.20    0 1 1 2 .97599 .97599 28.39 0.0   0.0   0        1 84.0     0    0     0  57 0.00   1    1 0          0            0       0       0  0
% load dimSizes from par file (probably only works for 2D acquisitions)
fid                         = fopen(parFile);
iRowFirstImageInformation   = 101;
C                           = textscan(fid, '%s','delimiter', '\n');

header.FOV_mm                      = cell2mat(textscan(C{1}{31}, ...
    '.    FOV (ap,fh,rl) [mm]                :   %f%f%f'));
header.offcenter_mm                = cell2mat(textscan(C{1}{34}, ...
    '.    Off Centre midslice(ap,fh,rl) [mm] :   %f%f%f'));
header.angulation_deg              = cell2mat(textscan(C{1}{33}, ...
    '.    Angulation midslice(ap,fh,rl)[degr]:   %f%f%f'));
header.TR_s                        = 1e-3*cell2mat(textscan(C{1}{30}, ...
    '.    Repetition time [msec]             :   %f'));

%% read data from first image information row
par = str2num(C{1}{iRowFirstImageInformation});
header.xDim = par(10);
header.yDim = par(11);
header.zDim = str2num(C{1}{22}(regexp(C{1}{22},'[\d]')));
header.tDim = str2num(C{1}{23}(regexp(C{1}{23},'[\d]')));

header.sliceOrientation    = par(26); % 1 = tra, 2 = sag, 3 = cor
header.sliceThickness      = par(23);
header.sliceGap            = par(24);

header.xres = par(29);
header.yres = par(30);
header.zres = header.sliceThickness + header.sliceGap;

%% Read additional info from whole data matrix
parAllRows = cell2mat(cellfun(@str2num, ...
    C{1}(iRowFirstImageInformation:end), 'UniformOutput', false));
%noOfImg = size(parAllRows,1);
header.nImageTypes = numel(unique(parAllRows(:,5)));
header.nEchoes = numel(unique(parAllRows(:,2)));


%% Read info for data rescaling
if header.nImageTypes < 2
    header.rescaleIntercept    = par(12);
    header.rescaleSlope        = par(13);
    header.scaleSlope          = par(14);
else
    for iIm = 1:header.nImageTypes
        par = str2num(C{1}{iRowFirstImageInformation+iIm});
        header.rescaleIntercept(iIm)    = par(12);
        header.rescaleSlope(iIm)        = par(13);
        header.scaleSlopeVec(iIm)       = par(14);
    end
end

fclose( fid );