function [dimInfoArray, sfxArray, selectionArray] = split(this, splitDims)
% splits dimInfo along specified dimensions, creating an array of reduced
% dimInfos and corresponding string suffixes for index-specific object naming
%
%   Y = MrDimInfo()
%   Y.split(splitDims)
%
% This is a method of class MrDimInfo.
%
% IN
%   splitDims   cell(1,nSplitDims) array of dimLabels or
%               [1,nSplitDims] vector of dim indices along which to split
% OUT
%   dimInfoArray
%               cell(nElementsSplitDim1, ..., nElementsSplitDimN) of
%               dimInfos, split along splitDims, i.e. containing one
%               element along these dimensions only
%   sfxArray    cell(nElementsSplitDim1, ..., nElementsSplitDimN) of
%               string suffixes indicating labels of elements of
%               dimInfoArray via their selected relative indices along the
%               split dimensions, e.g.
%               _t0005_dr0003_echo0004
%   selectionArray
%               cell(nElementsSplitDim1, ..., nElementsSplitDimN) of
%               selections, defined as cells containing propertyName/value
%               pairs over split dimensions, e.g.
%               {'t', 5, 'dr', 3, 'echo', 4}
%
% EXAMPLE
%   split
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-09-22
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% verify valid input
isValidSplitDim = ~isempty(splitDims) && ...
    ((isnumeric(splitDims) && all(splitDims <= this.nDims)) || ...
    (ischar(splitDims) && any(ismember(this.dimLabels, splitDims))) || ...
    (iscell(splitDims) && all(cellfun(@ischar, splitDims)) && any(ismember(this.dimLabels, splitDims))));

if nargin < 2 || isempty(splitDims) || ~isValidSplitDim % no splitting
    dimInfoArray = {this.copyobj};
    sfxArray = {''};
    selectionArray = {[]};
    
else % split along specified dimensions
    
    iSplitDims = this.get_dim_index(splitDims); % transform label string to numerical index
    nSplitDims = numel(iSplitDims);
    
    nSamplesInSplit = this.nSamples(iSplitDims);
    
    if nSplitDims == 1 % special case to create (N,1) cell
        dimInfoArray = cell(nSamplesInSplit, 1);
    else
        dimInfoArray = cell(nSamplesInSplit);
    end
    sfxArray = dimInfoArray; % copy of same size, still empty
    selectionArray = dimInfoArray; % copy of same size, still empty
    
    nSelections = numel(dimInfoArray);
    
    % {1...nSplitDim1, 1...nSplitDim2, ..., 1...nSplitDimN}
    indexPerDimArray = cellfun(@(x) 1:x, num2cell(nSamplesInSplit), ...
        'UniformOutput', false);
    % make index grid of all possible combinations of indices in splitDim
    splitIndexGrid = cell(1, nSplitDims);
    [splitIndexGrid{:}] = ndgrid(indexPerDimArray{:});
    dimLabelsSplit = this.dimLabels(iSplitDims);
    
    for iSelection = 1:nSelections
        % construct selection cell dimLabel, index,
        %   e.g. {'t', 5, 'dr', 3, 'echo', 4}
        selection = {};
        sfxArray{iSelection} = '';
        for iDim = 1:nSplitDims
            dimLabel = dimLabelsSplit{iDim};
            dimIndex = splitIndexGrid{iDim}(iSelection);
            selection = {selection{:}, dimLabel, dimIndex};
            sfxArray{iSelection} = sprintf('%s_%s%04d', sfxArray{iSelection}, ...
                dimLabel, dimIndex);
        end
        selectionDimInfo = this.select(selection{:});
        selectionArray{iSelection} = selection;
        dimInfoArray{iSelection} = selectionDimInfo;
    end
    
end
