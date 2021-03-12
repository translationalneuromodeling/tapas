function filenameArray = get_filename(this, varargin)
% returns the full filename as given in this.parameters.save, an additional
% prefix or suffix can be added;
% NOTE: For high-dimensional nifti data, an array of filenames is given that
% corresponds to the filenames created by MrDataNd.split when saving to
% disk.
%
%
%   Y = MrImage()
%   filenameArray = Y.get_filename(this, 'prefix', 'raw', 'isSuffix', false, ...
%                   'isMixedCase', 1, 'splitDims', 'unset');
%
% This is a method of class MrImage.
%
% IN
%   prefix          add prefix to filename
%                   default: none
%
%   isSuffix        prefix is added as suffix instead
%                   default: false
%
%   isMixedCase     ensures mixed case for pre-/suffixes
%                   default: true
%
%   splitDims       allows to retrieve filenames for individual splits for
%                   nD images (as they are used in MrImage.save)
%                   - default: {} only filename root is returned
%                   - {}: only root of filename (w/o high dim suffixes) is
%                         returned, irrespective of dimensionality of
%                         MrDimInfo
%                   - 'unset': filenames for the defaults split dims as
%                              defined in MrImage.save are returned, i.e. everything
%                              other than {'x', 'y', 'z', 't'}
%                   - {'sD1', 'sD2'}: filenames for split along dimension 
%                               'sD1', 'sD2' etc are returned
%
% OUT
%
% EXAMPLE
%   filename = Y.get_filename;
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-12
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%
defaults.prefix = '';
defaults.isSuffix = 0;
defaults.isMixedCase = [];
defaults.splitDims = {};

args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

% default setting for isMixedCase: use it, if prefix was specified!
if isempty(isMixedCase)
    isMixedCase = ~isempty(prefix);
end

% parse splitDims
tmpFileName = fullfile(this.parameters.save.path, ...
    this.parameters.save.fileName);
[~, ~, ext] = fileparts(tmpFileName);

% for 4D nifti images or n-dim non-nifti images
% don't split and don't specify suffix, keep root filename
if isequal(splitDims, 'unset')
    switch ext
        case {'.nii', '.img'}
            splitDims = 5:this.dimInfo.nDims;
        otherwise
            splitDims = [];
    end
end

% create individual file names for each saved split
[~, sfxArray] = this.dimInfo.split(splitDims);
nSelections = numel(sfxArray);
filenameArray = cell(size(sfxArray));
for iSelection = 1:nSelections
    % create new sub-directory for raw data to store it there temporarily
    switch prefix
        case 'dimInfo'
            [~,fn,~] = fileparts(this.parameters.save.fileName);
            filename = fullfile(this.parameters.save.path, [fn '_dimInfo.mat']);
        case 'dimInfoRaw'
            [~,fn,~] = fileparts(this.parameters.save.fileName);
            filename = fullfile(this.parameters.save.path, 'raw', [fn '_dimInfo.mat']);
        case 'raw'
            filename = fullfile(this.parameters.save.path, ...
                prefix, this.parameters.save.fileName);
        otherwise
            % create filename
            filename = fullfile(this.parameters.save.path, ...
                this.parameters.save.fileName);
            % prefix filename
            filename = tapas_uniqc_prefix_files(filename, prefix, isSuffix, isMixedCase);
    end
    filename = tapas_uniqc_prefix_files(filename, sfxArray{iSelection}, 1, 0);
    filenameArray{iSelection} = filename;
end

if nSelections == 1
    filenameArray = filenameArray{1};
end