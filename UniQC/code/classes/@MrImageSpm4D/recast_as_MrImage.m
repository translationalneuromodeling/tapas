function recastMrImage = recast_as_MrImage(this)
% recasts an MrImageSpm4D as a 4D MrImage, changes available methods etc.
%
%   Y = MrImageSpm4D()
%   YasMrImage = Y.recast_as_MrImage()
%
% This is a method of class MrImageSpm4D.
%
% IN
%
% OUT
%
% EXAMPLE
%   recast_as_MrImage
%
%   See also MrImageSpm4D

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-03
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


recastMrImage = MrImage();
recastMrImage.update_properties_from(this);

% house keeping: rename, if default name was used before, add info
% about recast
if strcmp(recastMrImage.name, 'MrImageSpm4D')
    recastMrImage.name = 'MrImage';
end
recastMrImage.info{end+1,1} = 'recast_as_MrImage';

