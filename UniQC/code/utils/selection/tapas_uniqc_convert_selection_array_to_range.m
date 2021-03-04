function selectionIndexRangeCell = tapas_uniqc_convert_selection_array_to_range(...
    selectionIndexArrayCell)
% Converts array of single element selection cells (name/value pairs) into
% single cell of name / range pairs
%
%   selectionIndexRangeCell = tapas_uniqc_convert_selection_array_to_range(selectionIndexArrayCell)
%
% IN
%   selectionIndexArrayCell     cell(nValuesDim1,...,nValuesDim1) of 
%                               dimLabel / dimValue pairs as used in
%                               MrDimInfo.split (selectionArray)
%                               e.g., 
%                               {'coils', 1, 'echo', 1}, ..., {'coils', 1, 'echo', 3}
%                               ...
%                               {'coils', 8, 'echo', 1}, ..., {'coils', 8, 'echo', 3}
%
% OUT
%   selectionIndexRangeCell     cell(1,2*dimLabels) of dimLabel /
%                               dimValueRange pairs, 
%                               e.g., {'coils', 1:8, 'echo', 1:3}
%
%
% EXAMPLE
%   selectionIndexArrayCell = tapas_uniqc_convert_selection_range_to_array(...
%       {'coils', 1:8, 'echo', 1:3});
%   selectionIndexRangeCell = tapas_uniqc_convert_selection_array_to_range(...
%       selectionIndexArrayCell); % should return {'coils', 1:8, 'echo', 1:3}
%
%   See also tapas_uniqc_convert_selection_range_to_array MrDimInfo.split

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


% use 1st as template, assuming they all share the same structure
selectionIndexRangeCell = selectionIndexArrayCell{1};
nDims = numel(selectionIndexRangeCell)/2;

nElements = numel(selectionIndexArrayCell);
for iDim = 1:nDims
    % collect all indices from all elements, then reduce to unique once
    % since on ndgrid, a lot of index pairs share elements...
    selectionIndexRangeCell{2*iDim} = zeros(1,nElements);
    for iElement = 1:nElements
        selectionIndexRangeCell{2*iDim}(iElement) = ...
            selectionIndexArrayCell{iElement}{2*iDim};
    end
    selectionIndexRangeCell{2*iDim} = sort(unique(selectionIndexRangeCell{2*iDim}));
end
