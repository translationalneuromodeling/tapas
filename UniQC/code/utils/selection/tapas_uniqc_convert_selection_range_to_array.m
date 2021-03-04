function selectionIndexArrayCell = tapas_uniqc_convert_selection_range_to_array(selectionIndexRangeCell)
% Converts selection name/range pairs to cells of single selections
%
%   selectionIndexArrayCell = tapas_uniqc_convert_selection_range_to_array(selectionIndexRangeCell)
%
% IN
%   selectionIndexRangeCell     cell(1,2*dimLabels) of dimLabel /
%                               dimValueRange pairs, 
%                               e.g., {'coils', 1:8, 'echo', 1:3}
%
% OUT
%   selectionIndexArrayCell     cell(nValuesDim1,...,nValuesDim1) of 
%                               dimLabel / dimValue pairs as used in
%                               MrDimInfo.split (selectionArray)
%                               e.g., 
%                               {'coils', 1, 'echo', 1}, ..., {'coils', 1, 'echo', 3}
%                               ...
%                               {'coils', 8, 'echo', 1}, ..., {'coils', 8, 'echo', 3}
%
% EXAMPLE
%   selectionIndexArrayCell = tapas_uniqc_convert_selection_range_to_array(...
%       {'coils', 1:8, 'echo', 1:3});
%   selectionIndexRangeCell = tapas_uniqc_convert_selection_array_to_range(...
%       selectionIndexArrayCell); % should return {'coils', 1:8, 'echo', 1:3}
%
%   See also tapas_uniqc_convert_selection_array_to_range MrDimInfo.split

% Author:   Lars Kasper
% Created:  2018-05-04
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


indexRanges = selectionIndexRangeCell(2:2:end);

nElementsPerDim = cellfun(@numel, indexRanges);
nDims = numel(nElementsPerDim);

selectionIndexArrayCell = cell(nElementsPerDim);

indexGrid = cell(1,nDims);
[indexGrid{:}] = ndgrid(indexRanges{:});

nElements = prod(nElementsPerDim);
for iElement = 1:nElements
    selectionIndexArrayCell{iElement} = cell(1,2*nDims);
    for iDim = 1:nDims
        selectionIndexArrayCell{iElement}{2*iDim-1} = ...
            selectionIndexRangeCell{2*iDim-1};
        selectionIndexArrayCell{iElement}{2*iDim} = ...
            indexGrid{iDim}(iElement);
    end
end