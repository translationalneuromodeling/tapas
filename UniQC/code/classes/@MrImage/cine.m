function filename = cine(this, varargin)
% Saves video of multidimensional dataset to an avi-file
% TODO: make it work with any plot-function and slider4D-save option
%
%
%   Y = MrImage()
%   Y.cine(inputs)
%
% This is a method of class MrImage.
%
% IN
%   given as property name/value pairs:
%
%           movieFormat      per default 'gif' for animated gif
%           pathSave    filename inclusive path of the animated gif
%           cineDim     plot the first two dimensions, along the
%                       third.
%           speed       in frames per second (default: 1)
%
% OUT
%           filename    filename of the saved animated image, per default
%                       file is saved in the current folder with a name
%                       refering to the input filename.
%
% EXAMPLE
%   cine
%
%   See also MrImage

% Author:   Laetitia Vionnet
% Created:  2015-04-17
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

defaults.movieFormat    = 'avi';
defaults.cineDim        = [1,2,3];
defaults.signalPart      = 'abs'; %'abs' or 'angle'
defaults.nColorsPerMap  = 65536;
defaults.displayRange   = [];
defaults.pathSave       = pwd;
defaults.filename       = [];
defaults.speed          = 1;
defaults.colorMap               = 'gray';
defaults.colorBar               = 'off';


args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

isGif = strcmpi(movieFormat, 'gif');


showColorbar = strcmpi(colorBar, 'on');

timeStamp = datestr(now, 'yyyymmdd_HHMMSS');

if isempty(filename)
    filename = [tapas_uniqc_str2fn(this.name), '_', timeStamp '.' movieFormat];
end

% re-arrange the data dimensions
dimData                 = size(this.data);
N                       = length(dimData);
dimDataOrder            = 1:N;
dimDataOrderNew         = 1:N;
permutations            = perms(dimDataOrder);

% find the first permutation that satisfies the cineDim order
for iCn = 1:length(permutations)
    if ~isequal(dimDataOrderNew(1:3),cineDim)
        dimDataOrderNew = permutations(iCn,:);
    end
end

% permute the data
DataPerm                = permute(this.data,dimDataOrderNew);

Data                    = DataPerm(:,:,:,1,1,1,1,1,1,1,1,1);


switch signalPart
    case 'all'
        % do nothing, leave dataPlot as is
    case 'abs'
        Data = abs(Data);
    case {'angle', 'phase'}
        Data = angle(Data) + pi;
    case 'real'
        Data = real(Data);
    case 'imag'
        Data = imag(Data);
end

nFrame                  = size(Data,3);


if isempty(displayRange)
    displayRange = [min(Data(:)), 0.8*max(Data(:))];
end


figure
% first frame
imagesc(Data(:,:,1));

axis tight equal
set(gca,'nextplot','replacechildren','visible','off')
caxis(displayRange);
if showColorbar
    colorbar; 
end
colormap(colorMap);

switch movieFormat
    case 'gif'
    case 'avi'
        profile = 'Uncompressed AVI';
    case 'mpeg'
        profile = 'MPEG-4';
end

if ~isGif
    myVideoWriter = VideoWriter(filename, profile);
    myVideoWriter.FrameRate = speed;
    open(myVideoWriter);
end

f = getframe;
[im,map] = rgb2ind(f.cdata,nColorsPerMap,'nodither');
im(1,1,1,nFrame) = 0;

for iSlice = 1:nFrame % number of frame of the animated gif
    imagesc(Data(:,:,iSlice));
    caxis(displayRange);
    if showColorbar
        colorbar;
    end

    f = getframe;
    
    if ~isGif
        myVideoWriter.writeVideo(f);
    end
    
    
    for iCn = 1:round(1/speed)
        im(:,:,1,iSlice) = rgb2ind(f.cdata,map,'nodither');
    end
    
end

if isGif
    imwrite(im,map,fullfile(pathSave,filename),'DelayTime',1/speed,'LoopCount',inf)
else
    close(myVideoWriter);
end

end