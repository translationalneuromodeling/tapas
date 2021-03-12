function varargout = split(this, varargin)
% Splits MrDataNd (or MrImage) along splitDims, sets filenames of created splitDataNd
% and optionally saves split images to file.
%
%   Y = MrDataNd()
%   [splitDataNd, selectionArray] = Y.split('splitDims', {'dimLabelSplit1, ..., dimLabelSplitN}, ...
%                           'doSave', false, 'fileName', newSplitFilename, ...
%                           'doRemoveDims', false)
%
% This is a method of class MrDataNd.
%
% IN
%   doRemoveDims    removes singleton dimensions from dimInfo after split
%                   (i.e. also the dimension information for dimensions
%                   along which split was performed)
%                   default: false
% OUT
%   splitDataNd cell(nElementsSplitDim1, ..., nElementsSplitDimN) of
%               MrDataNd, split along splitDims, i.e. containing one
%               element along these dimensions only
%   selectionArray
%               cell(nElementsSplitDim1, ..., nElementsSplitDimN) of
%               selections, defined as cells containing propertyName/value
%               pairs over split dimensions, e.g.
%               {'t', 5, 'dr', 3, 'echo', 4}
%
% EXAMPLE
%   split
%
%   See also MrDataNd MrDataNd.save

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-09-25
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


defaults.doSave = false;
defaults.fileName = this.get_filename(); % take only root of filename
defaults.splitDims = 'unset'; % changed below!
defaults.doRemoveDims = false;

args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

% defaults splitDims are adapted depending on file extension to have
% e.g. default 4D nifti files.
[fp, fn, ext] = fileparts(fileName);
if isequal(splitDims, 'unset')
    switch ext
        case {'.nii', '.img'}
            % TODO: decide whether other 4th dimension could be saved into
            % a nifti, e.g., TE for multi-echo data
            splitDims = setdiff(1:this.dimInfo.nDims, this.dimInfo.get_dim_index({'x','y','z','t'}));
            
        otherwise
            splitDims = [];
    end
end

  
% suppress output of mkdir when existing is better than "if exist",
% because the latter will also
% return true if relative directory exists anywhere else on path
if doSave
    [~,~] = mkdir(fp);
end

% 1. create all selections,
% 2. loop over all selections
%       a) to select sub-image
%       b) to adapt name of subimage with selection suffix
%       c) to save (with extension-specific) single-file save

[dimInfoArray, sfxArray, selectionArray] = this.dimInfo.split(splitDims);


nSelections = numel(dimInfoArray);

if nargout
    splitDataNd = cell(size(dimInfoArray));
end

for iSelection = 1:nSelections
    tempDataNd = this.select(selectionArray{iSelection});
    tempDataNd.parameters.save.path = fp;
    saveFileName = [fn sfxArray{iSelection} ext];
    tempDataNd.parameters.save.fileName = saveFileName;
    saveFileNameArray{iSelection} = fullfile(fp, saveFileName);
    
    if doRemoveDims
        tempDataNd.remove_dims([]);
    end
    
    if doSave
        tempDataNd.write_single_file();
    end
    
    if nargout
        splitDataNd{iSelection} = tempDataNd;
    end
    
end

if nargout
    varargout{1} = splitDataNd;
end

if nargout > 1
    varargout{2} = selectionArray;
end

if nargout > 2
    varargout{3} = saveFileNameArray;
end


