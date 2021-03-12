function [handleImageArray, nameImageArray] = get_all_image_objects(this, type)
% Returns cell of handles & names for all/selected image objects in MrSeries
%
%   Y = MrSeries()
%  [handleImageArray, nameImageArray] = Y.get_all_image_objects(inputs)
%
% This is a method of class MrSeries.
%
% IN
%
% OUT
%   handleImageArray    cell(nImages, 1) of handles to MrImages
%   nameImageArray      cell(nImages, 1) of names of properties of MrSeries
%                       which are MrImages
%
% EXAMPLE
%   get_all_image_objects
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-09
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


%handleImageArray = findall(this, 'MrImage');

if nargin < 2
    type = 'all';
end

nameStatImageArray = {'mean'; 'sd'; 'snr'; 'coeffVar'; 'diffLastFirst'; 'diffOddEven'};
nameDataImageArray = {'data'};
nameAnatomy = {'anatomy'};

switch lower(type)
    case 'data'
        nameImageArray = nameDataImageArray;
    case {'stat', 'stats', 'statistical'}
        nameImageArray = nameStatImageArray;
    case 'anatomy'
        nameImageArray = nameAnatomy;
    case 'all'
        nameImageArray = [nameDataImageArray; nameStatImageArray; ...
            nameAnatomy];
end

nImages = numel(nameImageArray);

for iImage = 1:nImages
    handleImageArray{iImage} = this.(nameImageArray{iImage});
end