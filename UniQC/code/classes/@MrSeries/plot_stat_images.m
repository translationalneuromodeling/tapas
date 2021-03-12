function this = plot_stat_images(this, varargin)
% plots statistical images (mean/snr/sd/diffLastFirst) in comprehensive
% plot for several slices
%
%   Y = MrSeries()
%   Y.plot_stat_images('ParameterName', ParameterValue)
%
% This is a method of class MrSeries.
%
% IN
%   'ParameterName'
%   'selectedSlices     [1, nSlices]  vector of slice indices to be
%                                     plotted. typically 3 (low, middle,
%                                     high)
%   'maxSnr'            maximum SNR to be plotted in colorscale
%   'maxSignal'         maximum signal in image
%
% OUT
%
% EXAMPLE
%   plot_stat_images
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-06
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

defaults.selectedSlices = round(...
    linspace(3,this.data.geometry.nVoxels(3) - 2 , 3));
defaults.statImageArray = {'mean', 'snr', 'sd', 'diffLastFirst', 'diffOddEven'};
defaults.maxSnr = max(this.snr.data(:));
defaults.maxSignal = max(this.mean.data(:));
args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

nImages = numel(statImageArray);
nSlices = numel(selectedSlices);

stringTitle = sprintf('%s - stat_images', this.name);
fh = figure('Name', stringTitle);
set(fh, 'WindowStyle', 'docked');


% colorbar axes with reasonable scaling

mostValuesPositive = sum(this.mean.data(:) < -0.1) < 100;

if mostValuesPositive % range of mean and SNR values only positive?
    relInterval = [0 1];
else
    relInterval = [-1 1];
end

cax = ...
    [
    maxSignal * relInterval
    maxSnr * relInterval
    maxSignal/maxSnr*3 * [0 1] 
    round(.02*maxSignal*[-1 1])
    round(.02*maxSignal*[-1 1])
    ];

% starting with Matlab 2019b, allows for tighter subplots
hasTiledLayout = exist('tiledlayout'); 

if hasTiledLayout
    tiledlayout(nSlices, nImages,'TileSpacing','Compact','Padding','Compact');
end

for row = 1:nSlices
    slice = selectedSlices(row);
    for col = 1:nImages
        img = statImageArray{col};
        if hasTiledLayout
            nexttile
            hs(row, col) = gca;
        else
            hs(row, col) = subplot(nSlices, nImages, nImages*(row-1) + col);
        end
        imagesc(this.(img).data(:,:,slice));
        axis square; %axis off;
        caxis(cax(col,:));
        
        % nice legend
        if row == 1
            title(img)
        end
        if col == 1
            ylabel(sprintf('slice %d', slice));
        end
        if row==nSlices
            colorbar('horiz');
        end
    end
end
if exist('suptitle', 'builtin'), suptitle(tapas_uniqc_str2label(stringTitle)); end
