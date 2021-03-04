function [outputImage, varargout] = apply_spm_method_per_4d_split(this, methodHandle, varargin)
% Applies SPM-related method of MrImageSpm4D to a higher-dimensional MrImage ...
% splitting data into 4D chunks executing SPM method for each split separately
% before recombining the image
%
%   Y = MrImage()
%   [newY, outVar] = Y.apply_spm_method_per_4d_split(this, methodHandle, ...
%                       'paramName', paramValue, ...)
%
% This is a method of class MrImage.
%
% NOTE:     Splitting into 4D MrImage is always performed on all but
%           {'x','y','z','t'} dimensions
%
% IN
%   methodHandle    string or @handle to method of MrImageSpm4D to be
%                   executed
%
%   propName/Value pairs:
%
%   methodParameters    extra parameters to be passed on to methodHandle
%                       beside the MrImageSpm4D
%   splitDimLabels      default: all but {'x','y','z',t'}
%
% OUT
%
% EXAMPLE
%   apply_spm_method_per_4d_split
%
%   See also MrImage

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

args = tapas_uniqc_propval(varargin, defaults);

tapas_uniqc_strip_fields(args);
%% create 4 SPM dimensions via complement of split dimensions
% if not specified, standard dimensions are taken
if isempty(splitDimLabels)
    dimLabelsSpm4D = {'x','y','z','t'};
else
    dimIndexSpm4D = setdiff(1:this.dimInfo.nDims, ...
        this.dimInfo.get_dim_index(splitDimLabels));
    
    % error, if split into 4D would not work...
    if numel(dimIndexSpm4D) > 4
        error('tapas:uniqc:MrImage:SplitDimensionsLargerThan4D', ...
            'Specified split dimensions do not split into 4 (or less) dimensional images');
    else
        dimLabelsSpm4D = this.dimInfo.dimLabels(dimIndexSpm4D);
    end
end

%% one-on-one (estimation-application per split item)
% simplest case, split and apply one by one
imageArray = this.split_into_MrImageSpm4D(dimLabelsSpm4D);

nSplits = numel(imageArray);

imageArrayOut = cell(size(imageArray));
% prepare output container with right size
if nargout > 1 
    varargout = cell(nSplits,nargout-1);
end
for iSplit = 1:nSplits
    if nargout < 2
        imageArrayOut{iSplit} = methodHandle(imageArray{iSplit}, ...
            methodParameters{:});
    else % allow passing of additional output arguments (e.g. realignment parameters)
        [imageArrayOut{iSplit}, varargout{iSplit,:}] = methodHandle(imageArray{iSplit}, ...
            methodParameters{:});
    end
end

% only recast if only one image
if numel(imageArrayOut) == 1
    outputImage = imageArrayOut{1}.recast_as_MrImage();
else
    outputImage = imageArrayOut{1}.combine(imageArrayOut);
end