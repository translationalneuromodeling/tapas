function this = load(this, inputDataOrFile, varargin)
% loads (meta-)data from file(s), order defined by loopDimensions
%
%   Y = MrDataNd()
%   Y.load(varargin)
%
% This is a method of class MrDataNd.
%
% IN
%   inputDataOrFile     can be one of the following inputs
%                       1)  a Matrix: MrDataNd is created along with a
%                           dimInfo matching the dimensions of the data
%                           matrix
%                       2)  a figure/axes handle: MrDataNd tries to infer
%                           from the plot type and data, what Image shall
%                           be loaded from the specified handle
%                       3)  a file-name: MrDataNd is loaded from the
%                           specified file
%                       4)  cell(nFiles,1) of file names to be concatenated
%                       5)  a directory: All image files in the specified
%                           directory
%                       6)  a regular expression for all file names to be
%                           selected
%                           e.g. 'folder/fmri.*\.nii' for all nifti-files
%                           in a folder
%
%
%   varargin:   propertyName/value pairs, referring to
%               a) loading of files, e.g. 'updateProperties' or
%               'selectedVolumes'
%               b) 'select' struct to select a subset of data
%               c) 'dimInfo' object
%               d) property/value pairs for dimInfo
%               e) 'affineTrafo' object
%
%
%
% OUT
%   this        MrDataNd with updated .data and .dimInfo
%   affineTransformation
%               For certain file types, the affineTransformation is saved as a
%               header information. While ignored in MrDataNd, it might be
%               useful to return it for specific processing
%               See also MrImage MrAffineTransformation
%
% EXAMPLE
%   Y.load(gca)
%   Y.load(gcf);
%   Y.load(figure(121)); % to make distinction between fig handle and 1-element integer image
%
%   See also MrDataNd demo_save MrDataNd.read_matrix_from_workspace MrDataNd.read_data_from_graphics_handle

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-10-21
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% update/load path for data and dimInfo
% In load:
% --------
% Files are determined and loop over individual files is started.
%   In read_single_file:
% ----------------------
%   1:  Values are derived from the input matrix (nSamples) and/or file
%       (header info).
%       DimInfo is initiated here.
%   2:  If a _dimInfo.mat file exists, this is automatically loaded as well.
%       DimInfo properties are updated.
% End of read_single_file.
% ------------------------
% Single files are combined.
% 3:  If a dimInfo/affineTrafo object is an input argument,
%     dimInfo/affineTrafo properties are updated.
% 4:  If prop/val pairs are given,
%     dimInfo/affineTrafo properties are updated.

%% 0. Preliminaries
% process input parameters
if nargin < 2
    inputDataOrFile = this.get_filename();
end
defaults.select = [];
defaults.dimInfo = [];
defaults.affineTransformation = [];
[args, argsUnused] = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

% Harvest propval for dimInfo constructor
[propValDimInfo, argsUnusedAfterDimInfo] = this.dimInfo.get_struct(argsUnused);

% input arguments for dimInfo constructor that are not properties
% themselves also need to be respected...
extraArgsDimInfo = this.dimInfo.get_additional_constructor_inputs();
[propValDimInfoExtra, argsUnusedAfterDimInfo] = tapas_uniqc_propval(argsUnusedAfterDimInfo, extraArgsDimInfo);

% create propValAffineTrafo
affineTrafo = MrAffineTransformation();
[propValAffineTransformation, loadInputArgs] = affineTrafo.get_struct(argsUnusedAfterDimInfo);
% check inputs
hasInputDimInfo = ~isempty(dimInfo);
hasInputAffineTransformation = ~isempty(affineTransformation);
hasPropValDimInfo = any(structfun(@(x) ~isempty(x), propValDimInfo));
hasPropValDimInfoExtra = any(structfun(@(x) ~isempty(x), propValDimInfoExtra));
hasPropValAffineTransformation = any(structfun(@(x) ~isempty(x), propValAffineTransformation));

hasSelect = ~isempty(select);
doLoad = 1;
%% 1. Determine files (for wildcards or folders)
isMatrix = isnumeric(inputDataOrFile) || islogical(inputDataOrFile);
isFigureOrAxesHandle = isa(inputDataOrFile, 'matlab.ui.Figure') || ...
    isa(inputDataOrFile, 'matlab.graphics.axis.Axes');
if isMatrix
    this.read_matrix_from_workspace(inputDataOrFile);
elseif isFigureOrAxesHandle
    this.read_data_from_graphics_handle(inputDataOrFile);
else % files or file pattern or directory
    isExplicitFileArray = iscell(inputDataOrFile) && ischar(inputDataOrFile{1});
    
    if isExplicitFileArray
        fileArray = inputDataOrFile;
    else
        fileArray = tapas_uniqc_get_filenames(inputDataOrFile);
    end
    
    % remove _dimInfo.mat from fileArray list
    [~, fileArray] = tapas_uniqc_find_info_file(fileArray, '_dimInfo.mat');
    %  get extra dimInfos from file names for select
    dimInfoExtra = MrDimInfo();
    dimInfoExtra.set_from_filenames(fileArray);
    % remove singleton dimensions
    dimInfoExtra.remove_dims();
    % now use select to only load subset of files
    % split select to within file (x, y, z, t) or between files
    [~, selectBetweenFiles, selectInFile] = dimInfoExtra.select(select);
    
    % number of files
    nFiles = numel(fileArray);
    
    %% 2. Load individual files into array of MrDataNd (including data of MrDimInfo)
    if nFiles == 1
        % only one file, read_single_files does everything that's necessary
        imgRead = read_single_file(this, fileArray{1}, loadInputArgs{:});
        % apply general select here (no splitting necessary)
        this.update_properties_from(imgRead.select(select));
    else
        % loop over nFiles and load each individually
        % initialize dataNdArray
        dataNdArray = cell(nFiles, 1);
        
        % actual constructor of possible sub-class of MrDataNd is
        % used here in loop, to reuse load as is in subclasses
        handleClassConstructor = str2func(class(this));
        
        for iFile = 1:nFiles
            % get filename
            fileName = fileArray{iFile};
            % get dimLabels from file name
            [dimLabels, dimValues] = tapas_uniqc_get_dim_labels_from_string(fileName);
            % check if dimLabels could be inferred from filename
            hasFoundDimLabelInFileName = ~isempty(dimLabels);
            if hasSelect
                % check if dimLabels and dimValues of this file are part of
                % selectDimInfo, to avoid unnecessary loading
                doLoad = all(cellfun(@ismember, num2cell(dimValues), selectBetweenFiles'));
            end
            if doLoad
                % load file into new file
                fprintf('Loading File %d/%d\n', iFile, nFiles);
                
                dataNdArray{iFile} = handleClassConstructor(fileName, ...
                    'select', selectInFile, loadInputArgs{:});
                % generate additional dimInfo
                if hasFoundDimLabelInFileName
                    % add units as samples
                    [units(1:numel(dimLabels))] = {'samples'};
                else
                    % generate generic dimLabels
                    dimLabels = {'file'};
                    dimValues = iFile;
                    units = 'sample';
                end
                % check if dimLabels already read
                hasDimLabel = any(ismember(dimLabels, dataNdArray{iFile}.dimInfo.dimLabels));
                if ~hasDimLabel
                    % add dimLabel and dim Value
                    dimsToAdd = dataNdArray{iFile}.dimInfo.nDims+1:dataNdArray{iFile}.dimInfo.nDims+numel(dimLabels);
                    dataNdArray{iFile}.dimInfo.add_dims(dimsToAdd, ...
                        'dimLabels', dimLabels, 'samplingPoints', dimValues,...
                        'units', units);
                end
            end
        end
        %% 3. Use combine to create one object
        % remove all empty cells from dataNdArray
        dataNdArray(cellfun(@isempty, dataNdArray)) = [];
        % use combine to create composite image
        imagesCombined = dataNdArray{1}.combine(dataNdArray);
        % add data to this
        this.update_properties_from(imagesCombined);
    end
end
% update dimInfo using input dimInfo
if hasInputDimInfo
    this.dimInfo.update_and_validate_properties_from(dimInfo);
end

% update dimInfo using prop/val dimInfo
if hasPropValDimInfo
    this.dimInfo.update_and_validate_properties_from(propValDimInfo);
end

if hasPropValDimInfoExtra
    this.dimInfo.set_dims(1:this.dimInfo.nDims, propValDimInfoExtra);
end


% update affineTransformation using input affineTransformation
if hasInputAffineTransformation
    this.affineTransformation.update_properties_from(affineTransformation);
end

% update affineTransformation using prop/val affineTransformation
if hasPropValAffineTransformation
    % affine matrix is a dependent property - cannot be changed via
    % update_properties_from
    if ~isempty(propValAffineTransformation.affineMatrix)
        this.affineTransformation.update_from_affine_matrix(propValAffineTransformation.affineMatrix)
    end
    this.affineTransformation.update_properties_from(propValAffineTransformation);
end
end

