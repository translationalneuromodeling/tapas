function [outputImage, outputParameters] = apply_spm_method_on_many_4d_splits(this, ...
    methodHandle, representationIndexArray, varargin)
% Applies SPM-related method of MrImageSpm4D to a higher-dimensional MrImage ...
% using representational 4D images as input to SPM to execute
% the method, runs a related method using the output parameters on the
% specified subsets of the MrImage
%
%   Y = MrImage()
%   [newY, outputParameters] = Y.apply_spm_method_on_many_4d_splits(this, ...
%                   methodHandle, representationIndexArray, ...
%                   'paramName', paramValue, ...)
%
% This is a method of class MrImage.
%
% Use case examples:
% 1) Realigning the first echo of a multi-echo dataset, and applying
%    the realignmnent to all echoes
% 2) Realigning the magnitude images and applying it to corresponding 
%    phase images
% 3) Realigning Sum-of-squares of all coil elements, and applying it to all
%    individual coils
%
%
% NOTE:     Splitting into 4D MrImage is per default performed on all but
%           {'x','y','z','t'} dimensions
%
% IN
%   methodHandle
%                   function handle to method of MrImageSpm4D to be
%                   executed for parameter estimation
%   representationIndexArray
%                   either cell(nRepresentations,1) of MrImageSpm4D, on
%                   on which methodHandle should be executed individually
%                   OR
%                   cell(nRepresentations,1) of cells with selection
%                   dimLabel/dimValue pairs
%                   e.g., {{'coil', 1, 'echo',1}; ...; {'coil', 1,
%                   'echo',3}}
%                   if each echo shall be realigned separately
%                   NOTE: a single representation can be given as one
%                   selection or one image, w/o extra cell brackets
%   
%   property Name/Value pairs:
%
%   methodParameters
%                   additional input parameters to that method
%   idxOutputParameters
%                   default: 1
%                   indices of output arguments of methodHandle that shall
%                   be transfered to applicationMethod
%   applicationIndexArray
%                   indices referring to the data chunks in representationIndexArray
%                   that shall be transformed by the same mapping SPM
%                   estimated for the those representatives
%                   NOTE: a single application can be given as one
%                   selection w/o extra cell brackets
%   applicationMethodHandle
%                   method which is applied on all elements of 
%                   applicationIndexArray using the nOutputArguments of the methodHandle
%                   executions
%   splitDimLabels  default: all but {'x','y','z',t'}
%
% OUT
%   outputParameters cell(nRepresentations, nOutputParameters)
%                   for each iteration of the method (i.e., one per
%                   specified representation), all defined output
%                   parameters (by idxOutputParameters) are stored here
%
% EXAMPLE
%   apply_spm_method_on_many_4d_splits
%
%   See also MrImage MrImage.realign MrImage.apply_spm_method_per_4d_split

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-22
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

defaults.methodParameters = {};
defaults.splitDimLabels = {};
defaults.idxOutputParameters = 1;
defaults.applicationIndexArray = {};
defaults.applicationMethodHandle = {};

args = tapas_uniqc_propval(varargin, defaults);

tapas_uniqc_strip_fields(args);

% single selection for representation (one selection or one MrImage)?
% i.e., not a cell of cells, e.g., {'coil', 1:8}, and not a cell of MrImages
isSingleSelection =  ~isempty(representationIndexArray) && ...
    ( isa(representationIndexArray, 'MrImage') || ... % a single image!
    (iscell(representationIndexArray) && ... % or a cell
    ~isa(representationIndexArray{1}, 'MrImage') && ... but not a cell or MrImages
    ~iscell(representationIndexArray{1})) ); % and not a cell of cells

if isSingleSelection
    representationIndexArray = {representationIndexArray};
end


%% create 4 SPM dimensions via complement of split dimensions
% if not specified, standard dimensions are taken
if isempty(splitDimLabels)
    dimLabelsSpm4D = {'x','y','z','t'};
else
    dimIndexSpm4D = setdiff(1:this.dimInfo.nDims, ...
        this.dimInfo.get_dim_index(splitDimLabels));
    
    % error, if split into 4D would not work...
    if numel(dimIndexSpm4D) ~= 4
        error('tapas:uniqc:MrImage:SplitDimensionsNot4D', ...
            'Specified split dimensions do not split into 4D images');
    else
        dimLabelsSpm4D = this.dimInfo.dimLabels(dimIndexSpm4D);
    end
end

%% one-on-many (estimation/application)
nRepresentations = numel(representationIndexArray);
outputParameters = cell(nRepresentations,numel(idxOutputParameters));
outputParametersTmp = cell(1,max(idxOutputParameters)); % container for return variables per representation run

imageArrayOut = cell(nRepresentations,1);
% empty applicationIndices in .select will select all data,
% and a split into all 4D subsets will be performed before application
if isempty(applicationIndexArray)
    % one empty selection (as cell) per representation
    applicationIndexArray = repmat({cell(1,1)}, nRepresentations, 1);
else
    % not a cell of cells, e.g., {'coil', 1:8}
    isSingleSelection = iscell(applicationIndexArray) && ~iscell(applicationIndexArray{1});
    if isSingleSelection
        applicationIndexArray = {applicationIndexArray};
    end
end

% Loop to run first SPM method (methodHandle) for all specified
% representational 4D images
for iRepresentation = 1:nRepresentations
    representationIndex = representationIndexArray{iRepresentation};
    
    % already images given (e.g. after previous math operations) for estimation,
    % no selection necessary
    if isa(representationIndex, 'MrImage')
        representationImage = representationIndex.recast_as_MrImageSpm4D();
    else
        representationImage = this.select(representationIndex{:}).recast_as_MrImageSpm4D();
    end
    
    % get output parameters for the estimation of this representation (image)...
    
    [outputParametersTmp{:}] = methodHandle(representationImage, methodParameters{:});
    outputParameters{iRepresentation,:} = outputParametersTmp{idxOutputParameters};
    
    % ...and apply these to all listed 4D sub-parts of the image, after
    % splitting into them
    applicationIndex = applicationIndexArray{iRepresentation};
    
    imageArrayApplication = ...
        this.select(applicationIndex{:}).split_into_MrImageSpm4D(dimLabelsSpm4D);
    
    nApplications = numel(imageArrayApplication);
    imageArrayOut{iRepresentation} = cell(nApplications,1);
    for iApplication = 1:nApplications
        imageArrayOut{iRepresentation}{iApplication} = ...
            applicationMethodHandle(imageArrayApplication{iApplication}, ...
            outputParameters{iRepresentation, :});
    end
end
% make cell of cell into nRepresentations*nApplications cell and combine
imageArrayOut = vertcat(imageArrayOut{:});
outputImage = imageArrayOut{1}.combine(imageArrayOut);

% cast 1x1 outputParameters cell back into one return argument
if numel(outputParameters) == 1
    outputParameters = outputParameters{1};
end
end