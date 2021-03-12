classdef MrDimInfo < MrCopyData
    % Holds all dimensionality info (r.g. range/dimLabels/units) of multidimensional data
    %
    %   See also demo_dim_info MrImage.select
    
    % Author:   Saskia Bollmann & Lars Kasper
    % Created:  2016-01-23
    % Copyright (C) 2016 Institute for Biomedical Engineering
    %                    University of Zurich and ETH Zurich
    %
    % This file is part of the TAPAS UniQC Toolbox, which is released
    % under the terms of the GNU General Public License (GPL), version 3.
    % You can redistribute it and/or modify it under the terms of the GPL
    % (either version 3 or, at your option, any later version).
    % For further details, see the file COPYING or
    %  <http://www.gnu.org/licenses/>.
    
    % the following properties can be fully derived from sampling points, ...
    % but are stored for convenience
    properties (Dependent)
        nDims;      % number of dimensions in dataset, default: 6
        
        nSamples;   % vector [1,nDims] of number of elements per dimension
        
        % vector [1,nDims] of resolutions for each dimension, ...
        % i.e. distance (in specified units) of adjacent elements,
        % NaN for non-equidistant spacing of elements
        % Example: TE = 2, 20, 35 ms => resolution = NaN
        resolutions;
        
        % vector [2,nDims] of [firstSample; lastSample] for each dimension ...
        ranges;
        
    end % properties (dependent)
    
    properties
        % cell(1,nDims) of string dimLabels for each dimension
        % default: {'x', 'y', 'z', 't', 'coil, 'echo'}
        dimLabels = {};
        
        % cell(1,nDims) of strings describing unit; '' for unit-free dims
        % default: {'mm', 'mm', 'mm', 's', '', 'ms'};
        units = {};
        
        % cell(1,nDims) of sampling position vectors for each dimension
        % Note: The entries for each dimension have to be numerical
        samplingPoints = {};
        
        % [1,nDims] vector of sampling width vectors for each dimension ...
        % Note: Typically, this will correspond to diff(samplingPoints);
        % However, if sampling does not cover the full interval between
        % consecutive points, it should be noted here
        % Example:  If z-coordinate has slice thickness and slice gap
        %           samplingWidths = sliceThickness
        %           resolutions = sliceThickness + sliceGap
        samplingWidths = [];
        
    end % properties
    
    
    
    methods
        
        function this = MrDimInfo(varargin)
            % Constructor of class, call via MrDimInfo('propertyName', propertyValue, ...
            % ...) syntax
            %               OR
            % MrDimInfo(fileName)
            %
            %               OR
            % MrDimInfo(fileArray) - additional dims are added via filename
            %
            %               OR
            % MrDimInfo(dimInfoStruct) - MrDimInfo object is costructed
            % from struct with same fields (especially for load and save)
            %
            % See also MrDimInfo.set_dims
            %
            % IN
            %   varargin    PropertyName/Value-pairs of MrDim-Info properties to be
            %               changed, e.g. resolutions, nSamples, units etc.
            %
            %   Properties:
            %
            %   'units'           cell(1,nDims) of strings describing unit; '' for unit-free dims
            %	'dimLabels'       cell(1,nDims) of string dimLabels for each changed dimension
            %
            %
            %   (1): 1st variant: explicit setting of sampling points for dimension(s)
            %
            %   'samplingPoints'  cell(1,nDims) of index vectors for each dimension
            %
            %
            %   (2)-(6): Other variants depend on setting some of the following parameters
            %
            %   'nSamples'              [1, nDims] number of samples per dimension
            %   'ranges'                [2, nDims] first and last sample per dimension
            %   'resolutions'           [1, nDims] resolution (in units), i.e. spacing
            %                           between sampling points per dimension
            %
            %   'arrayIndex'            index of samplingPoint to be set
            %   'samplingPoint'         value of sampling point at position arrayIndex
            %
            %   'firstSamplingPoint'    special case of samplingPoint, arrayIndex = 1 set
            %   'lastSamplingPoint'     special case of samplingPoint, arrayIndex = end set
            %
            %   Variants:
            %       (2) nSamples + ranges: sampling points computed automatically via
            %               samplingPoint(k) = ranges(1) + (ranges(2)-ranges(1))/nSamples*(k-1)
            %       (3) nSamples + resolutions + samplingPoint + arrayIndex:
            %               from samplingPoints(arrayIndex) = samplingPoints, others
            %               are constructed via
            %               [...    samplingPoint-resolution
            %                       samplingPoint
            %                       samplingPoint+resolution ...]
            %               until nSamples are created in total.
            %       (4) nSamples + resolutions + firstSamplingPoint:
            %               as (3), assuming arrayIndex = 1
            %       (5) nSamples + resolutions + lastSamplingPoint:
            %               as (3), assuming arrayIndex = end
            %       (6) nSamples Or resolution Or (samplingPoint+arrayIndex)
            %               missing input value taken from defaults
            %               nSamples        -> resolutions = 1, 1st sample 1
            %               resolutions     -> nSamples = 2, 1st sample 1
            %
            %           and always samplingWidths = resolution
            
            if nargin == 1 % single file, file array or struct is given
                if isstruct(varargin{1}) % is struct
                    this.update_properties_from(varargin{1});
                else % is file/file array
                    fileInput = varargin{1}; % extract input
                    isSingleFile = 0;
                    isFile = 1;
                    % check whether single filename or folder is given
                    if ischar(fileInput)
                        % determine whether file or folder
                        [~, ~, ext] = fileparts(fileInput);
                        if ~isempty(ext) % single file
                            fileName = fileInput;
                            isSingleFile = 1;
                        else
                            % or folder
                            fileInput = cellstr(spm_select('FPList',...
                                fileInput, '^*.*'));
                        end
                    elseif iscell(fileInput) && numel(fileInput) == 1
                        % fileArray with only one entry
                        fileName = fileInput{1}; % extract from cell array
                        isSingleFile = 1;
                    elseif isa(varargin{1}, 'MrAffineTransformation')
                        this.set_from_affine_geometry(varargin{1});
                        isFile = 0;
                    end
                    
                    if isFile
                        if isSingleFile
                            % load from single file
                            this.load(fileName);
                        else
                            % fileArray is given
                            % load dimInfo from first file in file array
                            this.load(fileInput{1});
                            this.set_from_filenames(fileInput);
                        end
                    end
                end
            else % prop/value pairs given
                
                propertyNames = varargin(1:2:end);
                propertyValues = varargin(2:2:end);
                % Find nSamples property, and corresponding value to determine
                % dimension
                iArgNsamples = tapas_uniqc_find_string(propertyNames, 'nSamples');
                iArgSamplingPoints = tapas_uniqc_find_string(propertyNames, 'samplingPoints');
                iArgRanges = tapas_uniqc_find_string(propertyNames, 'ranges');
                
                hasNsamples = ~isempty(iArgNsamples);
                hasExplicitSamplingPoints = ~isempty(iArgSamplingPoints);
                hasRanges = ~isempty(iArgRanges);
                
                if hasExplicitSamplingPoints
                    nDims = numel(propertyValues{iArgSamplingPoints});
                elseif hasNsamples
                    % otherwise, allow empty constructor for copyobj-functionality
                    nDims = numel(propertyValues{iArgNsamples});
                elseif hasRanges
                    nDims = numel(propertyValues{iArgRanges})/2;
                else
                    % guessed number of update dimensions
                    % find shortest given input to dimInfo and determine
                    % dimensionality from that
                    nDims =[];
                    for p = 1:numel(propertyNames)
                        nDims(p) = numel(propertyValues{p});
                    end
                    nDims = min(nDims);
                    
                    % populate nDims with non-samplingpoint parameters by adding dimensions
                    % , s.t. samplingPoints remains an empty cell for each
                    % dimension
                    % only samplingWidhts, units and dimLabels could be
                    % in varargin, so OK to add
                    this.add_dims(1:nDims, varargin{:});
                    
                end
                
                % allows empty constructor for copyobj
                if ~isempty(nDims)
                    this.set_dims(1:nDims, varargin{:});
                end
            end
            
        end
        % NOTE: Most of the methods are saved in separate function.m-files in this folder;
        %       except: constructor, delete, set/get methods for properties.
        %
        
        function nDims = get.nDims(this)
            nDims = numel(this.nSamples);
        end
        
        function this = set.nDims(this, nDimsNew)
            % Sets number of dimensions of dimInfo, via add_dims or ...
            % remove_dims
            nDimsOld = this.nDims;
            
            if nDimsNew > nDimsOld
                this.add_dims((nDimsOld+1):nDimsNew);
            elseif nDimsNew < nDimsOld
                this.remove_dims((nDimsNew+1):nDimsOld);
            end
        end
        
        function this = set.nSamples(this, nSamplesNew)
            % Changes nSamples by keeping given resolution and adding samples ...
            % at end of samplingPoints-vectors
            % TODO: Can we do this via set_dims? Or is this a problem for
            % empty dimInfos which are just created?
            % No: more for when we don't have a defined resolution
            % (non-equidistant!)
            
            nSamplesOld = this.nSamples;
            if numel(nSamplesNew) ~= numel(nSamplesOld)
                error('tapas:uniqc:MrDimInfo:NDimsChanged', ...
                    'nDims cannot change via nSamples, use add_dims instead');
            end
            
            iChangedDims = find(nSamplesOld ~= nSamplesNew);
            
            for iDim = iChangedDims
                nOld = nSamplesOld(iDim);
                nNew = nSamplesNew(iDim);
                if nOld > nNew
                    this.samplingPoints{iDim}((nNew+1):end) = [];
                else
                    if nOld == 0 % no samples before, create all new
                        offset = 0;
                    else % start samples from last existing
                        offset = this.samplingPoints{iDim}(nOld);
                    end
                    
                    % set default resolution to 1 for adding samples
                    res = this.resolutions(iDim);
                    if isnan(res)
                        res = 1;
                    end
                    
                    this.samplingPoints{iDim}((nOld+1):nNew) = ...
                        offset + (1:(nNew-nOld))*res;
                end
            end
            
        end
        
        function nSamples = get.nSamples(this)
            if isempty(this.samplingPoints)
                nSamples = [];
            else
                nSamples = cell2mat(cellfun(@numel, this.samplingPoints, ...
                    'UniformOutput', false));
            end
        end
        
        function this = set.resolutions(this, resolutionsNew)
            % Changes resolutions by setting the new resolution via
            % changeDim, the first sample is preserved
            resolutionsOld = this.resolutions;
            
            if numel(resolutionsNew) ~= numel(resolutionsOld)
                error('tapas:uniqc:MrDimInfo:NDimsChanged', ...
                    'nDims cannot change via resolutions, use add_dims instead');
            end
            
            iChangedDims = find(~arrayfun(@isequaln, resolutionsOld, resolutionsNew));
            this.set_dims(iChangedDims, 'resolutions', resolutionsNew(iChangedDims));
        end
        
        function resolutions = get.resolutions(this)
            if isempty(this.samplingPoints)
                resolutions = [];
            else
                
                for iDim = 1:this.nDims
                    res = unique(diff(this.samplingPoints{iDim}));
                    switch numel(res)
                        case 0 % one element samplingPoints, take its value (?)
                            if ~isempty(this.samplingWidths) && ...
                                    numel(this.samplingWidths) >= iDim && ...
                                    ~isnan(this.samplingWidths(iDim))
                                resolutions(iDim) = this.samplingWidths(iDim);
                            else
                                resolutions(iDim) = NaN;
                            end
                            % resolutions(iDim) = this.samplingPoints{iDim};
                        case  1 % single resolution, return it
                            resolutions(iDim) = res;
                        otherwise % if no unique resolution,
                            % first check if within single floating precision,
                            % accept that as same!
                            if max(abs(diff(res))) < eps(single(1))
                                resolutions(iDim) = mean(res);
                            else
                                %  otherwise, really non-equidistant spacing, return NaN for this
                                %  dim
                                resolutions(iDim) = NaN;
                            end
                    end
                end
            end
        end
        
        function this = set.ranges(this, rangesNew)
            % Changes ranges by keeping given nSamples and adjusting ...
            % samplingPoints (i.e. spacing i.e. resolution)
            rangesOld = this.ranges;
            
            if numel(rangesNew) ~= numel(rangesOld)
                error('tapas:uniqc:MrDimInfo:NDimsChanged', ...
                    'nDims cannot change via ranges, use add_dims instead');
            end
            
            iChangedDims = union(find(rangesOld(1,:) ~= rangesNew(1,:)), ...
                find(rangesOld(2,:) ~= rangesNew(2,:)));
            
            this.set_dims(iChangedDims, 'ranges', rangesNew(:,iChangedDims), ...
                'nSamples', this.nSamples(iChangedDims));
        end
        
        function ranges = get.ranges(this)
            ranges = [first(this); last(this)];
        end
        
        function centerSamples = center(this, iDim)
            % return center sample for given dimension (i.e. index ceil(nSamples/2))
            if nargin < 2
                iDim = 1:this.nDims;
            end
            
            centerSamples = nan(1, numel(iDim));
            for d = iDim
                if ~isempty(this.samplingPoints{d})
                    centerSamples(find(d==iDim)) = this.samplingPoints{d}(...
                        ceil(this.nSamples(d)/2));
                end
            end
            
        end
        
        function firstSamples = first(this, iDim)
            if nargin < 2
                iDim = 1:this.nDims;
            end
            
            firstSamples = nan(1, numel(iDim));
            for d = iDim
                if ~isempty(this.samplingPoints{d})
                    firstSamples(find(d==iDim)) = this.samplingPoints{d}(1);
                end
            end
            
        end
        
        % return last sampling point for all or given dimensions
        function lastSamples = last(this, iDim)
            if nargin < 2
                iDim = 1:this.nDims;
            end
            
            lastSamples = nan(1,numel(iDim));
            for d = iDim
                if ~isempty(this.samplingPoints{d})
                    lastSamples(find(d==iDim)) = this.samplingPoints{d}(end);
                end
            end
        end
        
        function [iDim, isValidLabel] = get_dim_index(this, dimLabel, varargin)
            % return index of dimension(s) given by a dimLabel
            % IN
            %   dimLabel  dimension label string (or array of strings).
            %             or dimension number or cell of dim numbers (for
            %             compatibility)
            %   varargin
            %   'invert'    true or false (default)
            %               if true, all other indices not within dimLabel
            %               are returned
            %
            % OUT
            %   iDim            index of dimension with corresponding label
            %   isValidLabel    [nLabels,1] returns for each given label 1/0
            %                   i.e. whether it is indeed a label of dimInfo
            defaults.invert = false;
            args = tapas_uniqc_propval(varargin,defaults);
            if isnumeric(dimLabel) % (vector of) numbers
                iDim = dimLabel;
                % cell of numbers:
            elseif iscell(dimLabel) && isnumeric(dimLabel{1})
                iDim = cell2mat(dimLabel);
            else % string or cell of strings
                isExact = 1;
                iDim = tapas_uniqc_find_string(this.dimLabels, dimLabel, isExact);
                if iscell(iDim)
                    isValidLabel = ~cellfun(@isempty, iDim);
                    iDim = iDim(isValidLabel); % remove unfound dimensions;
                    iDim = cell2mat(iDim)';
                else
                    isValidLabel = ~isempty(iDim);
                end
            end
            if args.invert
                iDim = setdiff(1:this.nDims,iDim);
            end
        end
        
    end
    
end
