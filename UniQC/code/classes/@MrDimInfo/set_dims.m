function this = set_dims(this, iDim, varargin)
% Sets dimension properties for specific dimension(s)...
%
%   Y = MrDimInfo()
%   Y.set_dims(inputs)
%
% This is a method of class MrDimInfo.
%
% set_dims is versatile in guessing missing values, e.g. by estimating
% actual sampling points from altered nSamples and resolutions, or shifting
% sampling points by given specific sampling point. See also demo_dim_info for
% details
%
% IN
%   iDim        (vector of) dimension index to be changed (e.g. 3 for 3rd
%               dimension) or cell of strings of dimension names
%   varargin    PropertyName/Value-pairs of MrDim-Info properties to be
%               changed, e.g. resolutions, nSamples, units etc.
%
%   Properties:
%
%   'units'           cell(1,nDims) of strings describing unit; '' for unit-free dims
%	'dimLabels'       cell(1,nDims) of string dimLabels for each changed dimension
%   'samplingWidths'  [1,nDims] vector of numbers referring to the width that
%                     a sample covers (usually equivalent to resolution,
%                     unless you have gaps between samples)
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
%   'originIndex'           special case, in which the origin (i.e.
%                           samplingPoint value [0 0 ... 0] can be defined
%                           by its arrayIndex position (non-integer index
%                           allowed)
%
%   Variants:
%       (2) nSamples + ranges: sampling points computed automatically via
%               samplingPoint(k) = ranges(1) + (ranges(2)-ranges(1))/(nSamples-1)*k
%           Note:   If nSamples is omitted, samplingPoints = ranges is
%                   assumed
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
%               missing input value from variant (3)-(5) is taken from
%               existing entries in dimInfo
%               Note: in all these following cases, the volume center is
%               assumed to be sampling point [0,0,0], i.e., sampling points
%               will be set as [-range/2, -range/2+resolution, ... range/2]
%               with range = (nSamples-1)*resolution and
%                    resolution = 1 (if not set otherwise)
%
%               nSamples        -> resolution and first sampling point are used to
%                               create nSamples (equidistant)
%               resolutions      -> nSamples and first sampling point are used to
%                               create new sampling-point spacing
%               samplingPoint   -> nSamples and resolution are used to
%                               create equidistant spacing of nSamples around
%                               sampling point
%
%
% OUT
%
% EXAMPLE
%   set_dims
%
%   See also MrDimInfo demo_dim_info

% Author:   Lars Kasper
% Created:  2016-01-28
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% nothing to do here...
if isempty(iDim)
    return
end

isStringiDimInput = ischar(iDim) || (iscell(iDim) && ischar(iDim{1}));
if isStringiDimInput
    dimLabel = iDim;
    [iDim, isValidLabel] = this.get_dim_index(dimLabel);
    nDimsToSplitVarargin = numel(iDim);
elseif iscell(iDim)
    iDim = cell2mat(iDim);
end

nDimsToSet = numel(iDim);

% no difference between splitting and set dimensions, if dim indices are
% given directly, not via label strings
if ~isStringiDimInput
    nDimsToSplitVarargin = numel(iDim);
    isValidLabel = ones(1,nDimsToSplitVarargin);
end

% check if structure variable of properties is given, or
% property/name value pairs; if struct, then convert to
isStructPropval = isstruct(varargin{1});

% convert struct input to prop/val pair cell array
if isStructPropval
    doRemoveEmptyProps = 1;
    propvalArray = tapas_uniqc_struct2propval(varargin{1},doRemoveEmptyProps);
else
    propvalArray = varargin;
end

iValidLabel = find(isValidLabel);
callForMultipleDimensions = nDimsToSplitVarargin > 1;
if callForMultipleDimensions
    vararginDim = tapas_uniqc_split_propval(propvalArray, nDimsToSplitVarargin);
    % call dimension setting for each dimension individually
    % and with respective caller arguments
    for d  = 1:nDimsToSet
        this.set_dims(iDim(d), vararginDim{iValidLabel(d)}{:});
    end
    
elseif nDimsToSet==1 % no execution for empty dimensions
    
    % overwritten, only, if set; firstSamplingPoint etc.
    defaults = this.get_additional_constructor_inputs();
    
    defaults.units = [];
    defaults.dimLabels = [];
    defaults.samplingPoints = []; % direct input of sampling points for dimensions
    defaults.samplingWidths = [];
    defaults.ranges = [];
    defaults.nSamples = [];
    defaults.resolutions = [];
    
    args = tapas_uniqc_propval(propvalArray, defaults);
    
    %% convert cells to content of their first entry, if parameters were
    % given with an enclosing {}, as if for multiple dimensions
    props = fieldnames(args);
    for p = 1:numel(props)
        if iscell(args.(props{p})) && numel(args.(props{p})) == 1
            args.(props{p}) = args.(props{p}){1};
        end
    end
    
    tapas_uniqc_strip_fields(args);
    %% First the easy stuff: explicit updates (without difficult dependencies)
    % of dimLabels and units
    
    if ~isempty(units)
        this.units{iDim} = units;
    else
        % if nothing set in object before, have a default...
        if isempty(this.units) || numel(this.units) < iDim || isempty(this.units{iDim})
            this.units{iDim} = this.get_default_dim_units(iDim);
        end
    end
    
    if ~isempty(dimLabels)
        this.dimLabels{iDim} = dimLabels;
    else
        % if nothing set in object before, have a default...
        if isempty(this.dimLabels) || numel(this.dimLabels) < iDim || isempty(this.dimLabels{iDim})
            this.dimLabels{iDim} = this.get_default_dim_labels(iDim);
        end
    end
    
    %% Now the hardest part: Update samplingPoints
    
    % differentiate cases of varargin for different setting methods
    doChangeOrigin                      = ~isempty(originIndex);
    doSetDimByRangeOnly                 = ~isempty(ranges) ...
        && isempty(nSamples);
    doSetDimByNsamplesAndRange          = ~isempty(nSamples) ...
        && ~isempty(ranges);
    doChangeResolution                  = ~isempty(resolutions) ...
        && all(isfinite(resolutions)); % non NaNs and Infs for updating samples from resolutions
    doChangeNsamples                    = ~isempty(nSamples);
    hasFirstSamplingPoint               = ~isempty(firstSamplingPoint);
    hasLastSamplingPoint                = ~isempty(lastSamplingPoint);
    hasSamplingPointIndexPair           = (~isempty(samplingPoint) ...
        && ~isempty(arrayIndex));
    doChangeBySingleSamplingPoint       = hasFirstSamplingPoint ...
        || hasLastSamplingPoint ...
        || hasSamplingPointIndexPair;
    hasExplicitSamplingPointsProperty   = ~isempty(samplingPoints);
    doChangeSamplingPoints              = doSetDimByRangeOnly ...
        || doSetDimByNsamplesAndRange ...
        || doChangeResolution ...
        || doChangeNsamples ...
        || hasExplicitSamplingPointsProperty ...
        || doChangeOrigin ...
        || doChangeBySingleSamplingPoint;
    
    if doChangeSamplingPoints % false, if only labels, units or samplingWidths is changed
        
        if ~hasExplicitSamplingPointsProperty % otherwise, we are done already, and can set
            %% set_dims(iDim, ...
            % 'nSamples', nSamples, 'ranges', [firstSample, lastSample])
            if doSetDimByNsamplesAndRange
                samplingPoints = linspace(ranges(1), ranges(2), nSamples);
            elseif doSetDimByRangeOnly
                samplingPoints = [ranges(1), ranges(2)];
            else % all other cases depend (partly) on resolutions,
                % nSamples or specific reference sampling points
                
                % e.g. for changing one sampling point only, i.e.
                %shifting all sampling points
                % set_dims(iDim, 'arrayIndex', 3, 'samplingPoint', 24,
                % 'units', 'mm');
                if ~doChangeResolution
                    % default resolution: 1...only occurs, if no samplingsPoints
                    % given in object yet
                    if isempty(this.resolutions) || numel(this.resolutions) < iDim
                        resolutions = 1;
                    else
                        resolutions = this.resolutions(iDim);
                    end
                    
                end
                
                %% set_dims(iDim, 'resolutions', 3) OR ...
                % set_dims(iDim, 'resolutions', 3, 'nSamples', 100)
                % => will keep first Sample of iDim and extend by new
                % resolution (and nSamples, if changed)
                if ~doChangeNsamples
                    % two samples per dimension are needed to establish
                    % resolution!
                    if isempty(this.nSamples) || numel(this.nSamples) < iDim ...
                            || this.nSamples(iDim) == 0
                        nSamples = 2;
                    else
                        nSamples = this.nSamples(iDim);
                    end
                end
                
                % if no sampling point given keep origin
                % if it doesn't exist, set it to volume center
                if isempty(samplingPoint) && isempty(originIndex)
                    originIndex = this.get_origin(iDim);
                    hasValidOriginIndex = ~isempty(originIndex) && ...
                        isfinite(originIndex); % no nans/infs
                    if ~hasValidOriginIndex
                        if any(strcmp(this.dimLabels{iDim}, {'x', 'y', 'z', 'r', 'p', 's'}))
                            originIndex = (nSamples+1)/2 - 1;
                        else
                            originIndex = -1;
                        end
                    end
                    nSamplesBefore = originIndex;
                    % origin index is in nifti format, thus one lower than what we (and matlab) counts the samplingPoints
                    samplingPoint = -nSamplesBefore * resolutions;
                    arrayIndex = 1;
                end
                
                %% fix one sampling point, derive others via equidistant
                % spacing of resolution
                
                %% set_dims (iDim, ...
                % 'firstSamplingPoint', 4, 'resolutions', 3);
                
                % settings for special (first/last) sampling points
                if hasFirstSamplingPoint
                    samplingPoint = firstSamplingPoint;
                    arrayIndex = 1;
                end
                
                %% set_dims (iDim, ...
                % 'lastSamplingPoint', 4, 'resolutions', 3);
                if hasLastSamplingPoint
                    samplingPoint = lastSamplingPoint;
                    arrayIndex = nSamples;
                end
                
                if doChangeOrigin
                    % TODO: recalc for non-integer originIndex; maybe via this.resolutions VS resolutions?
                    samplingPoint = 0;
                    arrayIndex = originIndex;
                end
                
                
                %[...   samplingPoint-resolution
                %       samplingPoint
                %       samplingPoint+resolution ...]
                samplingPoints(arrayIndex) = samplingPoint;
                samplingPoints(1:arrayIndex-1) = samplingPoint - ...
                    resolutions*((arrayIndex-1):-1:1);
                samplingPoints((arrayIndex+1):nSamples) = samplingPoint + ...
                    resolutions*(1:(nSamples-arrayIndex));
            end
        end
        
        if iscell(samplingPoints)
            % via subsasgn dimInfo.z.samplingsPoints = ... or if non-numeric sampling points set
            this.samplingPoints(iDim) = samplingPoints;
        else
            this.samplingPoints{iDim} = samplingPoints;
        end
        
    end
    
    %% Medium tricky: Updating sampling widths
    
    % update sampling widths either from direct input or via resolutions;
    % If resolution is NaN, keep previous value
    if ~isempty(samplingWidths)
        this.samplingWidths(iDim) = samplingWidths;
    else
        % overwrite sampling widths by resolutions, but only, if nothing
        % sensible is in there
        
        isValidSamplingWidth = numel(this.samplingWidths) >= iDim && ...
            ~isnan(this.samplingWidths(iDim)) && ~isinf(this.samplingWidths(iDim));
        
        isValidExistingResolutions = numel(this.resolutions) >= iDim && ...
            ~isnan(this.resolutions(iDim)) && ~isinf(this.resolutions(iDim));
        
        isValidInputResolutions = numel(resolutions) >= iDim && ...
            ~isnan(resolutions(iDim)) && ~isinf(resolutions(iDim));
        
        if ~isValidSamplingWidth
            if isValidInputResolutions
                % use input resolution as width
                this.samplingWidths(iDim) = resolutions;
            elseif isValidExistingResolutions
                % use computed resolution as width
                this.samplingWidths(iDim) = this.resolutions(iDim);
            else
                % set non-existing sampling widths to NaN
                this.samplingWidths(iDim) = NaN;
            end
        end
        
    end
else
    error('tapas:uniqc:MrDimInfoSetDimsNonExistingDimension', ...
        'Dimension with label "%s" does not exist in %s dimInfo', dimLabel, ...
        inputname(1));
end

end