function this = init_processing_step(this, module, varargin)
% initializes next processing step by creating folders for version tracking,
% shuffling data, and updating processing parameters
%
%   MrSeries = init_processing_step(MrSeries, module)
%
% This is a method of class MrSeries.
%
% IN
%   module      'realign', 'smooth', ...
%
% OUT
%
%   side effects:
%   new folder (with current data):
%       dirObject/<nProcessingSteps+1>_moduleName
%   parameters.processingLog
%   nProcessingSteps
%
% EXAMPLE
%   init_processing_step
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-01
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.



% NOTE: for each new processing step added here, it has to be decided which
% (input, raw, unprocessed) files are saved additionally


itemsSave = this.parameters.save.items;
doSave = ~strcmpi(itemsSave, 'none');
doSaveRaw = ismember(itemsSave, {'all'});
doSaveNifti = ismember(itemsSave, {'nii', 'all', 'processed'});
doSaveObject = ismember(itemsSave, {'object', 'all', 'processed'});

% set file-saving behavior of MrImage to keep disk files
this.data.parameters.save.keepCreatedFiles = ...
    1 ; % keeps files here, cleanup will happen in finish_processing_step

pathSaveRoot = this.parameters.save.path;

% save initial, unprocessed data
isFirstProcessingStep = ~this.nProcessingSteps;
if isFirstProcessingStep && doSave
    dirProcessing = sprintf('%03d_%s', this.nProcessingSteps, 'unprocessed');
    pathProcessing = fullfile(pathSaveRoot, dirProcessing);
    mkdir(pathProcessing);
    this.data.parameters.save.path = pathProcessing;
    this.data.parameters.save.fileName = 'data.nii';
    
    % save data (MrImage file)
    if doSaveNifti
        this.data.save();
    end
    
    % strip and save object as well
    if doSaveObject
        fileObject = fullfile(pathProcessing, 'MrObject.mat');
        MrObject = this.copyobj('exclude', 'data'); % copies object without data
        save(fileObject, 'MrObject');
    end
end

% specify new directory to save data here
this.nProcessingSteps   = this.nProcessingSteps + 1;
dirProcessing           = sprintf('%03d_%s', this.nProcessingSteps, module);
pathProcessing          = fullfile(pathSaveRoot, dirProcessing);

this.processingLog{end+1,1} = dirProcessing;


% module-specific adaptations, e.g. data copying

% for all matlabbatches where additional spm output files are saved
hasMatlabbatch = ismember(module, this.get_all_matlabbatch_methods());

% for all matlabbatches, where data is needed as raw.nii before job start
doesNeedDataNifti = ismember(module, {'realign', 'smooth', ...
    'specify_and_estimate_1st_level'});


if doSave || hasMatlabbatch
    mkdir(pathProcessing);
end

if doesNeedDataNifti % data has to be written to disk before running spm_jobman, prepare file-names!
    this.data.parameters.save.path = pathProcessing;
end

switch module
    case 'specify_and_estimate_1st_level'
        % save raw data
        this.data.save;
        
    case 'analyze_rois'
        % dummy image for path transfer
        inputDummyImage = varargin{1};
        inputDummyImage.parameters.save.path = pathProcessing;
        
    case 'compute_masks'
        inputImages = varargin{1};
        nImages = numel(inputImages);
        
        % set paths, save raw input files and prefix file names with "mask"... for all input files
        for iImage = 1:nImages
            inputImages{iImage}.parameters.save.path = pathProcessing;
            inputImages{iImage}.parameters.save.fileName = ...
                tapas_uniqc_prefix_files(inputImages{iImage}.parameters.save.fileName, ...
                'mask', 0, 1);
            inputImages{iImage}.save;
        end
        
    case 'compute_stat_images'
        
        % determine output file names, e.g. mean.nii and path of processing
        % directory
        [handleImageArray, nameImageArray] = this.get_all_image_objects('stats');
        for iImage = 1:numel(handleImageArray)
            handleImageArray{iImage}.parameters.save.path = pathProcessing;
            handleImageArray{iImage}.parameters.save.fileName = ...
                [nameImageArray{iImage} '.nii'];
        end
        
    case 'compute_tissue_probability_maps'
        
        % adjust path of input image to make it save-able
        inputImage = varargin{1};
        inputImage.parameters.save.path = pathProcessing;
        inputImage.parameters.save.keepCreatedFiles = 'all';
        
    case 'coregister'
        transformedImage = varargin{1};
        equallyTransformedImages = varargin{2};
        inputImages = [{transformedImage};...
            equallyTransformedImages];
        
        % set save path
        nImages = numel(inputImages);
        for iImage = 1:nImages
            inputImages{iImage}.parameters.save.path = ...
                pathProcessing;
            inputImages{iImage}.parameters.save.keepCreatedFiles = 'all';
        end
        
    case 'realign'
        
    case 'smooth'
        
        % set file names and save path for statistical images
    case 't_filter'
        this.data.parameters.save.path = pathProcessing;
        this.data.parameters.save.keepCreatedFiles = 'all';
end

end
