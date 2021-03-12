function [biasFieldCorrected, varargout] = segment(this, varargin)
% Segments brain images using SPM's unified segmentation approach.
% This warps the brain into a standard space and segment it there using tissue
% probability maps in this standard space.
%
% Since good warping and good segmentation are interdependent, this is done
% iteratively until a good tissue segmentation is given by probality maps
% that store how likely a voxel is of a certain tissue type
% (either in native or standard space).
% Furthermore, a deformation field from native to standard space (or back)
% has then been found for warping other images of the same native space.
%
%   Y = MrImage()
%   [biasFieldCorrected, tissueProbMaps, deformationFields, biasField] = ...
%   Y.segment(...
%       'representationIndexArray', representationIndexArray, ...
%       'spmParameterName1', spmParameterValue1, ...
%       ...
%       'spmParameterNameN', spmParameterValueN)
%
% This is a method of class MrImageSpm4D.
%
% NOTE: If an nD image is given, then all dimension > 3  will be treated as
% channels.
%
% IN
%   tissueTypes         cell(1, nTissues) of strings to specify which
%                       tissue types shall be written out:
%                       'GM'    grey matter
%                       'WM'    white matter
%                       'CSF'   cerebrospinal fluid
%                       'bone'  skull and surrounding bones
%                       'fat'   fat and muscle tissue
%                       'air'   air surrounding head
%
%                       default: {'GM', 'WM', 'CSF'}
%
%   mapOutputSpace      'native' (default) or 'warped'/'mni'/'standard'
%                       defines coordinate system in which images shall be
%                       written out;
%                       'native' same space as image that was segmented
%                       'warped' standard Montreal Neurological Institute
%                                (MNI) space used by SPM for unified segmentation
%  deformationFieldDirection determines which deformation field shall be
%                       written out,if any
%                       'none' (default) no deformation fields are stored
%                       'forward' subject => mni (standard) space
%                       'backward'/'inverse' mni => subject space
%                       'both'/'all' = 'forward' and 'backward'
%  saveBiasField        0 (default) or 1
%  biasRegularisation   describes the amount of expected bias field
%                       default: 0.001 (light)
%                       no: 0; extremely heavy: 10
%  biasFWHM             full-width-at-half-maximum of the Gaussian
%                       non-uniformity bias field (in mm)
%                       default: 60 (mm)
%  fileTPM              tissue probablity maps for each tissue class
%                       default: SPM' TPMs in spm/tpm
%  mrfParameter         strenght of the Markov Random Field cleanup
%                       performed on the tissue class images
%                       default: 1
%  cleanUp              crude routine for extracting the brain from
%                       segmented images ('no', 'light', 'thorough')
%                       default: 'light'
%  warpingRegularization regularization for the different terms of the
%                       registration
%                       default: [0 0.001 0.5 0.05 0.2]
%  affineRegularisation regularisation for the initial affine registration
%                       of the image to the tissue probability maps (i.e.
%                       into standard space)
%                       for example, the default ICBM template are slighlty
%                       larger than typical brains, so greater zooms are
%                       likely to be needed
%                       default: ICBM spase template - European brains
%  smoothnessFwhm       fudge factor to account for correlation between
%                       neighbouring voxels (in mm)
%                       default: 0 (for MRI)
%  samplingDistance     approximate distance between sampled points when
%                       estimating the model parameters (in mm)
%                       default: 3
%
%   Parameters for high-dim application:
%
%   representationIndexArray:   a selection (e.g. {'t', 1}) which is then
%                               applied to obtain one nD image, where all
%                               dimensions > 3 are treated as additional
%                               channels
%                               default representationIndexArray: all
%                               dimensions not the imageSpaceDims
%   imageSpaceDims              cell array of three dimLabels defining the
%                               dimensions that define the physical space
%                               the image is in
%                               default imageSpaceDims: {'x', 'y', 'z'}
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
%   biasCorrected       bias corrected images
%   tissueProbMaps      (optional) cell(nTissues,1) of 3D MrImages
%                       containing the tissue probability maps in the
%                       respective order as volumes,
%   deformationFields   (optional) cell(nDeformationFieldDirections,1)
%                       if deformationFieldDirection is 'both', this cell
%                       contains the forward deformation field in the first
%                       entry, and the backward deformation field in the
%                       second cell entry; otherwise, a cell with only one
%                       element is returned
%   biasField           (optional) bias field
%
% EXAMPLE
% [biasFieldCorrected, tissueProbMaps, deformationFields, biasField] =
%   Y.segment();
%
% for 7T images stronger non-uniformity expected
% [biasFieldCorrected, tissueProbMaps, deformationFields, biasField] = ...
%   m.segment('biasRegularisation', 1e-4, 'biasFWHM', 18, ...
%   'cleanUp', 2, 'samplingDistance', 2);
%
%   See also MrImage spm_preproc MrImageSpm4D.segment
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-12-23
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% spm parameters (details above)
spmDefaults.tissueTypes = {'WM', 'GM', 'CSF'};
spmDefaults.mapOutputSpace = 'native';
spmDefaults.deformationFieldDirection = 'none';
spmDefaults.saveBiasField = 0;
spmDefaults.biasRegularisation = 0.001;
spmDefaults.biasFWHM = 60;
spmDefaults.fileTPM = [];
spmDefaults.mrfParameter = 1;
spmDefaults.cleanUp = 'light';
spmDefaults.warpingRegularization = [0 0.001 0.5 0.05 0.2];
spmDefaults.affineRegularisation = 'mni';
spmDefaults.smoothnessFwhm = 0;
spmDefaults.samplingDistance = 3;

[spmParameters, unusedVarargin] = tapas_uniqc_propval(varargin, spmDefaults);

% for split/apply functionality
methodParameters = {spmParameters};

% use cases: abs of complex, single index on many!
defaults.representationIndexArray   = {{}}; % default: use all
defaults.imageSpaceDims             = {};
defaults.splitComplex               = 'mp';

args = tapas_uniqc_propval(unusedVarargin, defaults);
tapas_uniqc_strip_fields(args);

% set imageSpaceDims
if isempty(imageSpaceDims)
    imageSpaceDims = {'x','y','z'};
end

% check whether real/complex
isReal = isreal(this);

if isReal
    splitComplexImage = this.copyobj();
else
    splitComplexImage = this.split_complex(splitComplex);
end

% prepare output container with right size
varargoutForMrImageSpm4D = cell(numel(representationIndexArray),nargout-1);
hasArgOut = nargout > 1;
hasBiasField = nargout > 3;
for iRepresentation = 1:numel(representationIndexArray)
    
    % apply selection
    inputSegment = splitComplexImage.select(...
        representationIndexArray{iRepresentation});
    
    % Merge all n>3 dims, which are not part of the representationIndexArray, into 4D array
    mergeDimLabels = setdiff(inputSegment.dimInfo.dimLabels, imageSpaceDims);
    % additional channels need to be in the t dimensions so they become part of
    % the same nifti file
    % empty mergeDimLabels just return the original object, e.g. for true 3D
    % images
    [mergedImage, newDimLabel] = ...
        inputSegment.merge(mergeDimLabels, 'dimLabels', 't');
    
    if hasArgOut
        [biasFieldCorrected{iRepresentation}, varargoutForMrImageSpm4D{iRepresentation, :}] = ...
            mergedImage.apply_spm_method_per_4d_split(@segment, ...
            'methodParameters', methodParameters);
    else
        biasFieldCorrected{iRepresentation} = mergedImage.apply_spm_method_per_4d_split(@segment, ...
            'methodParameters', methodParameters);
    end
    
    % un-do merge operation using combine
    if ~isempty(mergeDimLabels)
        % not necessary for 4D images - just reset dimInfo
        if numel(mergeDimLabels) == 1
            origDimInfo = inputSegment.dimInfo;
            biasFieldCorrected{iRepresentation}.dimInfo = origDimInfo;
            % also for the bias fields
            if hasBiasField
                varargoutForMrImageSpm4D{iRepresentation, 3}.dimInfo = origDimInfo;
            end
        else
            % created original dimInfo per split
            origDimInfo = inputSegment.dimInfo.split(mergeDimLabels);
            % un-do reshape
            split_array = biasFieldCorrected{iRepresentation}.split('splitDims', newDimLabel);
            split_array = reshape(split_array, size(origDimInfo));
            % add original dimInfo
            for nSplits = 1:numel(split_array)
                split_array{nSplits}.dimInfo = origDimInfo{nSplits};
            end
            % and combine
            biasFieldCorrected{iRepresentation} = split_array{1}.combine(split_array);
            
            % same for the bias fields
            if hasBiasField
                % un-do reshape
                split_array = varargoutForMrImageSpm4D{iRepresentation, 3}{1}.split('splitDims', newDimLabel);
                split_array = reshape(split_array, size(origDimInfo));
                % add original dimInfo
                for nSplits = 1:numel(split_array)
                    split_array{nSplits}.dimInfo = origDimInfo{nSplits};
                end
                varargoutForMrImageSpm4D{iRepresentation, 3} = {split_array{1}.combine(split_array)};
            end
        end
    end
    
    if ~isReal
        % un-do complex split
        biasFieldCorrected{iRepresentation} = biasFieldCorrected{iRepresentation}.combine_complex();
    end
    
    % add representation index back to TPMs and deformation field to
    % combine them later
    if ~isempty(representationIndexArray{iRepresentation})
        for nOut = 1:nargout-2
            addDim = varargoutForMrImageSpm4D{iRepresentation, nOut}{1}.dimInfo.nDims + 1;
            dimLabels = representationIndexArray{iRepresentation}{1};
            % only pick the first one
            samplingPoints = representationIndexArray{iRepresentation}{2}(1);
            for nClasses = 1:numel(varargoutForMrImageSpm4D{iRepresentation, nOut})
                varargoutForMrImageSpm4D{iRepresentation, nOut}{nClasses}.dimInfo.add_dims(...
                    addDim, 'dimLabels', dimLabels, ...
                    'samplingPoints', samplingPoints);
            end
        end
    end
end

% combine bias field corrected
biasFieldCorrected = biasFieldCorrected{1}.combine(biasFieldCorrected);

% combine varargout
for nOut = 1:nargout-1
    toBeCombined = varargoutForMrImageSpm4D(:, nOut);
    % the TPMs are cell-arrays of images, so we need to combine them per
    % tissue class
    for nClasses = 1:numel(toBeCombined{1})
        for nCells = 1:size(toBeCombined, 1)
            thisCombine(nCells) = toBeCombined{nCells}(nClasses);
        end
        combinedImage{nClasses, 1} = thisCombine{1}.combine(thisCombine);
    end
    varargout{1, nOut} = combinedImage;
    clear toBeCombined thisCombine combinedImage;
end

end


