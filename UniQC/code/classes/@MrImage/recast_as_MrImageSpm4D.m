function recastMrImageSpm4D = recast_as_MrImageSpm4D(this)
% recasts a 4D MrImage as MrImageSpm4D, enabling usage of SPM interfaces
%
%   Y = MrImage()
%   Yas4D = Y.recast_as_MrImageSpm4D(inputs)
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%
% EXAMPLE
%   recast_as_MrImageSpm4D
%
%   See also MrImage

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



is4D = numel(this.dimInfo.get_non_singleton_dimensions()) <= 4;
if is4D
    % check {'x','y','z'} are 1st 3 dimensions
    % warning, if not in this order
    if ~isequal(this.dimInfo.get_dim_index({'x','y','z'}), [1 2 3])
        
        dimLabelString = '';
        for iDim = 1:min(this.dimInfo.nDims, 3), ...
                dimLabelString = [dimLabelString ' ' this.dimInfo.dimLabels{iDim}];
        end
        warning('first 3 dimensions are labeled {%s}, but {x y z} is assumed', ...
           dimLabelString);
    end
    recastMrImageSpm4D = MrImageSpm4D();
    recastMrImageSpm4D.update_properties_from(this);
    
    % house keeping: rename, if default name was used before, add info
    % about recast
    if strcmp(recastMrImageSpm4D.name, 'MrImage')
        recastMrImageSpm4D.name = 'MrImageSpm4D';
    end
    recastMrImageSpm4D.info{end+1,1} = 'recast_as_MrImageSpm4D';
else
    error('tapas:uniqc:MrImage:recast:TooManyDimensions', ...
        'recast only possible for max. 4D MrImages. Use split_into_MrImageSpm4D instead');
end
end
           
