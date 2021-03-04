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
%   Y = MrImageSpm4D()
%   [biasFieldCorrected, tissueProbMaps, deformationFields, biasField] = ...
%   Y.segment(...
%       'spmParameterName1', spmParameterValue1, ...
%       ...
%       'spmParameterNameN', spmParameterValueN)
%
% This is a method of class MrImageSpm4D.
%
% NOTE: If a 4D image is given, the fourth dimension will be treated as
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
%   See also MrImage spm_preproc

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-08
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% parse defaults
defaults.tissueTypes = {'WM', 'GM', 'CSF'};
defaults.mapOutputSpace = 'native';
defaults.deformationFieldDirection = 'none';
defaults.saveBiasField = 0;
defaults.biasRegularisation = 0.001;
defaults.biasFWHM = 60;
defaults.fileTPM = [];
defaults.mrfParameter = 1;
defaults.cleanUp = 'light';
defaults.warpingRegularization = [0 0.001 0.5 0.05 0.2];
defaults.affineRegularisation = 'mni';
defaults.smoothnessFwhm = 0;
defaults.samplingDistance = 3;

% tapas_uniqc_propval
args = tapas_uniqc_propval(varargin, defaults);

biasFieldCorrected = this.copyobj();
% if certain ouput parameters are requested, the input parameters must
% request them

if nargout > 2
    if strcmp(args.deformationFieldDirection, 'none')
        args.deformationFieldDirection = 'forward';
    end
end
if nargout > 3
    args.saveBiasField = 1;
end

if numel(biasFieldCorrected.dimInfo.get_non_singleton_dimensions) > 3
    % save split image file for processing as nii in SPM
    pathRaw = fileparts(biasFieldCorrected.get_filename('prefix', 'raw'));
    % make split image to prevent accidential misspecifications in this
    splitImage = biasFieldCorrected.copyobj();
    splitImage.parameters.save.path = pathRaw;
    
    [~, ~, saveFileNameArray] = splitImage.split('splitDims', biasFieldCorrected.dimInfo.dimLabels{4}, 'doSave', 1);
    
    [~, splitFileName] = fileparts(saveFileNameArray{1});
    [~, thisFileName] = fileparts(this.parameters.save.fileName);
    splitSuffix = regexprep(splitFileName, thisFileName, '');
else
    saveFileNameArray{1} = biasFieldCorrected.get_filename('prefix', 'raw');
    biasFieldCorrected.save('fileName', saveFileNameArray{1});
    splitSuffix = '';
end
% get matlabbatch
args.saveFileNameArray = saveFileNameArray;
matlabbatch = biasFieldCorrected.get_matlabbatch('segment', args);

save(fullfile(biasFieldCorrected.parameters.save.path, 'matlabbatch.mat'), ...
    'matlabbatch');
spm_jobman('run', matlabbatch);

% clean up: move/delete processed spm files, load new data into matrix
varargout = cell(1,nargout-1);
[varargout{:}] = biasFieldCorrected.finish_processing_step('segment', ...
    args.tissueTypes, args.mapOutputSpace, ...
    args.deformationFieldDirection, splitSuffix);
end