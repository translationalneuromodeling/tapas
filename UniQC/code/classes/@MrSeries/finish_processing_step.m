function this = finish_processing_step(this, module, varargin)
% finishes current processing step by deleting duplicate data and storing
% results of processing step
%
%   MrSeries = finish_processing_step(MrSeries, module, varargin)
%
% This is a method of class MrSeries.
%
% IN
%   module      'realign', 'smooth', ...
%
% OUT
%
% EXAMPLE
%   finish_processing_step
%
%   See also MrSeries MrSeries.init_processing_step

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
% files are saved additionally or which temporary files can be deleted/renamed
itemsSave = this.parameters.save.items;
doSave = ~strcmpi(itemsSave, 'none');
doSaveRaw = ismember(itemsSave, {'all'});
doSaveNifti = ismember(itemsSave, {'nii', 'all', 'processed'});
doSaveObject = ismember(itemsSave, {'object', 'all', 'processed'});

% determine where and which files have been changed from input argument
if iscell(varargin{1});
    inputImage = varargin{1}{1};
else
    inputImage = varargin{1};
end

pathSave        = inputImage.parameters.save.path;
pathRaw         = fileparts(inputImage.get_filename('prefix', 'raw'));

% delete additional, processed files...
switch module
    
    case 'compute_masks'
        maskImages = varargin{1};
        nImages = numel(maskImages);
        
        filesMask = cell(nImages,1);
        for iImage = 1:nImages
            filesMask{iImage} = ...
                maskImages{iImage}.get_filename;
            if doSaveNifti
                maskImages{iImage}.save;
            end
        end
        
        filesProcessed = filesMask;
        
    case 'compute_stat_images'
        % file names and paths already given in init_processing_step
        if doSaveNifti
            handleImageArray = this.get_all_image_objects('stats');
            for iImage = 1:numel(handleImageArray)
                handleImageArray{iImage}.save;
                filesProcessed{iImage} = ...
                    handleImageArray{iImage}.get_filename;
            end
        end
        
    case 'compute_tissue_probability_maps'
        createdFields = varargin{1};
        inputImage = varargin{2};
        nImages = numel(createdFields);
        
        filesFieldImages = cell(nImages,1);
        for iImage = 1:nImages
            filesFieldImages{iImage} = fullfile(...
                pathSave, ...
                createdFields{iImage}.parameters.save.fileName);
        end
        
        fileRaw     = inputImage.get_filename('prefix', 'raw');
        fileSeg8    = regexprep(fileRaw, '\.nii$', '_seg8\.mat');
        
        filesProcessed 	= [
            {inputImage.get_filename}
            filesFieldImages
            {fileSeg8}
            ];
        
        
    case 'coregister'
        transformedImage = varargin{1};
        equallyTransformedImages = varargin{2};
        
        % set files for delete
        filesProcessed = {transformedImage.get_filename};
        nImages = numel(equallyTransformedImages);
        for iImage = 1:nImages
            filesProcessed{end+1} = equallyTransformedImages{iImage}.get_filename;
            % load additionally transformed images
            handleInputImages = this.find('MrImage', 'name', ...
                equallyTransformedImages{iImage}.name);
            nameTransformed = handleInputImages{1}.name;
            handleInputImages{1}.load(equallyTransformedImages{iImage}.get_filename);
            handleInputImages{1}.name = nameTransformed;
        end
        
        
    case 'realign' % load realignment parameters into object
        
        fileRealignmentParameters = regexprep( ...
            tapas_uniqc_prefix_files(inputImage.get_filename('prefix', 'raw'), ...
            'rp_'), '\.nii$', '\.txt') ;
        this.glm.regressors.realign = load(fileRealignmentParameters);
        
        movefile(fileRealignmentParameters, pathSave);
        
        
        
    case 'smooth'
        filesProcessed = inputImage.get_filename();
        
        
    case 't_filter'
        if doSaveNifti
            this.data.save();
        end
        
    case 'specify_and_estimate_1st_level'
        % load design matrix
        spmDirectory = fullfile(this.glm.parameters.save.path, ...
            this.glm.parameters.save.spmDirectory);
        SPM = load(fullfile(spmDirectory, 'SPM.mat'));
        this.glm.designMatrix = SPM.SPM.xX.nKX;          
end

% delete raw sub-folder of current processing step
if ~doSaveRaw && exist(pathRaw, 'dir')
    delete(fullfile(pathRaw, '*'));
    rmdir(pathRaw);
end

if ~doSave
    tapas_uniqc_delete_with_hdr(filesProcessed);
end

% strip object data and save ...
if doSaveObject
    fileObject = fullfile(pathSave, 'MrObject.mat');
    MrObject = this.copyobj('exclude', 'data'); % copies object without data
    save(fileObject, 'MrObject');
end