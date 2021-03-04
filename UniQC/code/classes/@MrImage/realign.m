function [realignedImage, realignmentParameters] = realign(this, varargin)
% Realigns n-dimensional image according to representative derived 4D image(s),
% and applies resulting realignment parameters to respective subsets of the
% n-d image
%
%   Y = MrImage()
%   rY = Y.realign('representationIndexArray', representationIndexArray, ...
%       'applicationIndexArray', applicationIndexArray, ...
%       'spmParameterName1', spmParameterValue1, ...
%       ...
%       'spmParameterNameN', spmParameterValueN)
%
% This is a method of class MrImage.
%
% IN
%
% most SPM realign est/reslice parameters, enforcing congruency between
% est/reslice and ignoring file naming options:
%
%   quality         0..1, estimation quality, share of voxels included in estimation
%                   default: 0.9
%   separation      separation distance (mm) between evaluated image points in estimation
%                   default: 4
%   smoothingFwhm   FWHM (mm) of Gaussian smoothing kernel used for estimation
%                   default: 5
%   realignToMean   boolean; if true, 2-pass procedure, registering to mean
%                   default: true
%   interpolation   degree of b-spline interpolation for estimation and reslicing
%                   default: 7
%   wrapping        fold-over direction (phase encode)
%                   default: [0 0 0] % none
%   weighting       weighting image for estimation
%                   can be filename or MrImage
%                   default: '' % none
%   masking         mask incomplete timeseries?
%                   default: true
%
%   Parameters for high-dim application:
%
%   representationIndexArray:   either an MrImageObject or a selection
%                               (e.g. {'echo', 1} which is then applied to
%                               obtain one 4D image
%                               default representationIndexArray: first
%                               index of all extra (non-4D) dimensions
%   applicationIndexArray:      a selection which defines one or multiple
%                               4D images on which the estimated parameters
%                               are applied
%                               default applicationIndexArray: all non-4D
%                               dimensions
%   splitComplex                'ri' or 'mp'
%                               If the data are complex numbers, real and
%                               imaginary or magnitude and phase are
%                               realigned separately.
%                               default: mp (magnitude and p)
%                               Typically, realigning the magnitude and
%                               applying it to the phase data makes most
%                               sense; otherwise, using real and imaginary
%                               part, more global phase changes would
%                               impact on estimation
%
% OUT
%   realigned MrImage object, estimated realignment parameters
%

% EXAMPLE
%
% % realign individual echoes based on the mean of all echoes
%   realignedImage = image.realign('representationIndexArray',...
%   image.mean('echo'), 'applicationIndexArray', {'echo', 1:10});
%
% % realign with modifications to SPM's defaults
%   Y = MrImage()
%   [rY, realignmentParameters] = Y.realign('quality', 0.99, ...
%       'smoothingFwhm', 2);
%
%   [rY, realignmentParameters] = Y.realign('quality', 0.99, ...
%       'weighting', weightingMrImage);
%
%   See also MrImage MrImage.wrap_spm_method MrImageSpm4D.realign
%   \demo\MrImage\demo_realign spm_realign spm_cfg_realign

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-21
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

realignedImage = this.copyobj;

% most SPM realign est/reslice parameters, enforcing congruency between
% est/reslice and ignoring file naming options
% See also spm_realign or spm_cfg_realign

spmDefaults.quality = 0.9;         % 0..1, estimation quality, share of voxels included in estimation
spmDefaults.separation = 4;        % separation distance (mm) between evaluated image points in estimation
spmDefaults.smoothingFwhm = 5;     % FWHM (mm) of Gaussian smoothing kernel used for estimation
spmDefaults.realignToMean = 1;     % boolean; if true, 2-pass procedure, registering to mean
spmDefaults.interpolation = 7;     % degree of b-spline interpolation for estimation and reslicing
spmDefaults.wrapping = [0 0 0];    % fold-over direction (phase encode)
spmDefaults.weighting = '';        % weighting image for estimation
spmDefaults.masking = 1;           % mask incomplete timeseries?

[spmParameters, unusedVarargin] = tapas_uniqc_propval(varargin, spmDefaults);

% for split/apply functionality
methodParameters = {spmParameters};

% use cases: abs of complex, single index on many!
defaults.representationIndexArray   = {}; % default: take first index of extra dimensions!
defaults.applicationIndexArray      = {}; % default: apply to all
defaults.splitDimLabels             = {};
defaults.splitComplex               = 'mp';
defaults.idxOutputParameters        = 2;
args = tapas_uniqc_propval(unusedVarargin, defaults);
tapas_uniqc_strip_fields(args);

% check whether (real/complex) 4D
nonSDims = realignedImage.dimInfo.get_non_singleton_dimensions();
is4D = numel(nonSDims) == 4;
isReal = isreal(realignedImage);
isReal4D = is4D && isReal;
isComplex = ~isReal;
if isReal4D % just do realign once!
    [realignedImage, realignmentParameters] = realignedImage.apply_spm_method_per_4d_split(@realign, ...
        'methodParameters', methodParameters);
else
    if isComplex
        realignedImage = realignedImage.split_complex(splitComplex);
        
        % for non-empty applicationIndexArray, we have to augment each cell
        % element by the new last dimension (real/imag part or abs/magn)
        % distinction, and application should be to both, therefore, choose
        % all samples:
        if ~isempty(applicationIndexArray)
            newLabels = realignedImage.dimInfo.dimLabels{end}; % complex_mp or complex_ri
            newSamples = realignedImage.dimInfo.samplingPoints{end}; % 1,2, i.e., both parts
            % append the new dimension to all selections in cell
            applicationIndexArray = append_dim_to_selections(applicationIndexArray, newLabels, newSamples);
            
        end
    end
    
    %% create 4 SPM dimensions via complement of split dimensions
    % if not specified, standard dimensions are taken
    if isempty(splitDimLabels)
        dimLabelsSpm4D = {'x','y','z','t'};
        splitDimLabels = setdiff(realignedImage.dimInfo.dimLabels, dimLabelsSpm4D);
    end
    
    % default representation: take first index of all extra (non-4D) dimensions
    % e.g., {{'coil'}    {[1]}    {'echo'}    {[1]}}
    % for 4D, splitDimLabels will be empty, representationIndex can stay
    % empty, because 4D split is clear
    if isempty(representationIndexArray)
        if ~isempty(splitDimLabels)
            representationIndexArray = reshape(splitDimLabels, 1, []);
            representationIndexArray(2,:) = {1};
            representationIndexArray = {reshape(representationIndexArray, 1, [])};
        end
    elseif isComplex && iscell(representationIndexArray)
        % update representationArray with new selection for first
        % representation as well, if it is a selection (and not an image
        % object)
        newLabels = realignedImage.dimInfo.dimLabels{end}; % complex_mp or complex_ri
        newSamples = realignedImage.dimInfo.samplingPoints{end}(1); % 1,2, i.e., both parts
        representationIndexArray = append_dim_to_selections(representationIndexArray, newLabels, newSamples);
    end
    
    [realignedImage, realignmentParameters] = ...
        realignedImage.apply_spm_method_on_many_4d_splits(@realign, ...
        representationIndexArray, ...
        'methodParameters', methodParameters, ..., ...
        'applicationIndexArray', applicationIndexArray, ...
        'applicationMethodHandle', @(x,y) apply_realign(x, y, methodParameters{:}), ...
        'idxOutputParameters', idxOutputParameters);
    
    if isComplex
        %% reassemble complex realigned images into one again
        realignedImage = realignedImage.combine_complex();
    end
end
end

% append the labels and sample indices of the new dimension at the end of
% each selection cell
function selectionArray = append_dim_to_selections(selectionArray, newLabels, newSamples)
% not a cell of cells, e.g., {'coil', 1:8}, only 1
% augmentation
isSingleSelection = iscell(selectionArray) && ~iscell(selectionArray{1});
if isSingleSelection
    selectionArray{end+1} = newLabels;
    selectionArray{end+1} = newSamples;
else
    % multiple applications, augment each with new dimension
    for iApplication = 1:numel(selectionArray)
        selectionArray{iApplication}{end+1} = newLabels;
        selectionArray{iApplication}{end+1} = newSamples;
    end
end
end
