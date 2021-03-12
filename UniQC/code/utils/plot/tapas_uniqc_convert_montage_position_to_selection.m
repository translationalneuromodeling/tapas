function newPosition = tapas_uniqc_convert_montage_position_to_selection(montagePosition, ...
    montageSize, dimInfoSelection, selectionIndexArray)
% Converts (mouse) position in montage into x,y,z selection of plotted data
%
%    newPosition = tapas_uniqc_convert_montage_position_to_selection(montagePosition, ...
%                   montageSize, dimInfoSelection))
%
% IN
%
%   dimInfoSelection    dimInfo of selected image part for montage plot
%   selectionIndexArray
%                       cell(nDims,1) of absolute indices that selected
%                       subpart for plotting had in full dataset
%                       2nd output argument of MrImage.select
% OUT
%
% EXAMPLE
%   tapas_uniqc_convert_montage_position_to_selection MrImage.select
%
%   See also
 
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-03-28
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

nX = dimInfoSelection.nSamples('x');
nY = dimInfoSelection.nSamples('y');
nZ = dimInfoSelection.nSamples('z');

montageX = montagePosition(1);
montageY = montagePosition(2);

% montageX and montageY have to be the swapped here, because 1st and second
% dimension of matlab array and displayed image are swapped as well
x = mod(round(montageY), nX);
y = mod(round(montageX), nY);

% final sample has to be set manually, because mod is between 0 and nX-1
if ~x, x = nX; end
if ~y, y = nY; end

nRows = montageSize(1);
nCols = montageSize(2);

iRow = min(max(1, ceil(montageY/nX)), nRows);
iCol = min(max(1, ceil(montageX/nY)), nCols);

iZ = min(max(1, (iRow-1)*nCols+iCol), nZ);

iDimZ = dimInfoSelection.get_dim_index('z');
z = selectionIndexArray{iDimZ}(iZ);
newPosition = [x y z];
