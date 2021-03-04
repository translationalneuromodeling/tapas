function varargout = finish_processing_step(this, module, varargin)
% finishes current processing step by deleting SPM-related temporary data
% on disk and writing back processed data from disk into data matrix
% results of processing step
%
%   Y = MrImage()
%   Y.finish_processing_step(module, varargin)
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%
% EXAMPLE
%   Y.finish_processing_step('smooth', fileOutputSpm);
%
%   See also MrImage

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-03
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


itemsSave = this.parameters.save.keepCreatedFiles;

switch itemsSave
    case 0
        itemsSave = 'none';
    case 1
        itemsSave = 'all';
end

doSave = ~strcmpi(itemsSave, 'none');
doSaveRaw = ismember(itemsSave, {'all'});
doSaveNifti = ismember(itemsSave, {'nii', 'all', 'processed'});
doSaveObject = ismember(itemsSave, {'object', 'all', 'processed'});
doSaveProcessed = ismember(itemsSave, {'processed'});

pathSave        = this.parameters.save.path;
fileRaw         = this.get_filename('prefix', 'raw');
pathRaw         = fileparts(fileRaw);
fileProcessed   = this.get_filename;
nameImage       = this.name;

% for file prefixes
isSuffix = false;
isMixedCase = true;
doKeepCreatedFiles = (this.parameters.save.keepCreatedFiles == 1) & ...
    ~strcmpi(this.parameters.save.keepCreatedFiles, 'none');
hasMatlabbatch = ismember(module, this.get_all_matlabbatch_methods());
varargout = {};

% nifti leftovers exist and resulting files have to be renamed
if hasMatlabbatch
    fileMatlabbatch = fullfile(this.parameters.save.path, ...
        'matlabbatch.mat');
    filesCreated = {
        fileRaw
        fileProcessed
        fileMatlabbatch
        };
    switch module
        case 'apply_transformation_field'
            prefixOutput = 'w';
            fileOutputSpm = tapas_uniqc_prefix_files(fileRaw, prefixOutput);
            
        case 'coregister_to'
            % has matlabbatch, but does not create unnecessary files...,
            % since matlabbatch not executed...
            prefixOutput = '';
            fileStationaryImage = varargin{1};
            tapas_uniqc_delete_with_hdr(fileStationaryImage);
            fileOutputSpm = varargin{2}; % SPM output of reslice
            filesCreated = [
                filesCreated
                {fileStationaryImage}
                ];
            
        case 'apply_realign'
            
            prefixOutput = 'r';
            fileOutputSpm = tapas_uniqc_prefix_files(fileRaw, prefixOutput);
            filesCreated = {
                fileRaw
                fileProcessed
                };
            
        case 'realign'
            
            fileWeightingRealign = varargin{1};
            
            prefixOutput = 'r';
            fileOutputSpm = tapas_uniqc_prefix_files(fileRaw, prefixOutput);
            
            fileRealignmentParameters = regexprep(...
                tapas_uniqc_prefix_files(fileRaw, 'rp_'), '\.nii', '\.txt');
            fileRealignMean = tapas_uniqc_prefix_files(fileRaw, 'mean');
            filesCreated = [
                filesCreated
                {fileRealignmentParameters; fileRealignMean;}
                fileWeightingRealign
                ];
            
            varargout{1} = load(fileRealignmentParameters, '-ascii');
            
        case 'reslice'
            
            prefixOutput = 'r';
            fileOutputSpm = tapas_uniqc_prefix_files(fileRaw, prefixOutput);
            
            fnTargetGeometry = varargin{1};
            
            % dummy image of target geometry always deleted
            tapas_uniqc_delete_with_hdr(fnTargetGeometry);
            
        case 'segment'
            % parse inputs
            tissueTypes = varargin{1};
            imageOutputSpace = varargin{2};
            deformationFieldDirection = varargin{3};
            splitSuffix = varargin{4};
            
            % the bias-corrected anatomical is the output
            prefixOutput = 'm';
            fileOutputSpm = tapas_uniqc_prefix_files(fileRaw, prefixOutput);
            tpmOutputSpm = MrImage(tapas_uniqc_prefix_files(fileOutputSpm, '*', 1), ...
                'updateProperties', 'save');
            delete(tapas_uniqc_prefix_files(fileOutputSpm, '*', 1));
            tpmOutputSpm.save();
            
            % get current and new tissue probability map file names
            allTissueTypes = {'GM', 'WM', 'CSF', 'bone', 'fat', 'air'};
            indTissueTypes = find(ismember(lower(allTissueTypes), ...
                lower(tissueTypes)));
            nTissues = numel(indTissueTypes);
            
            filesTpm = cell(nTissues,1);
            filesTpmProcessed = cell(nTissues,1);
            
            for iTissue = 1:nTissues
                indTissue = indTissueTypes(iTissue);
                filesTpm{iTissue} = tapas_uniqc_prefix_files(fileRaw, ...
                    sprintf('c%d%', indTissue));
                % add wildcard for multi-channel segmentation
                filesTpm{iTissue} = tapas_uniqc_prefix_files(filesTpm{iTissue}, splitSuffix, 1);
                filesTpmProcessed{iTissue} = tapas_uniqc_prefix_files( ...
                    fullfile(this.parameters.save.path, sprintf('%s.nii', ...
                    lower(allTissueTypes{indTissue}))), ...
                    'tissueProbabilityMap', isSuffix, isMixedCase);
            end
            
            if ismember(imageOutputSpace, {'mni', 'standard', 'template', 'warped'})
                filesTpm = tapas_uniqc_prefix_files(filesTpm, 'w');
                filesTpmProcessed = tapas_uniqc_prefix_files(filesTpmProcessed, ...
                    'warped', isSuffix, isMixedCase);
            end
            
            % deformation field file names, if saved
            filesDeformationField = {};
            filesDeformationFieldProcessed = {};
            
            hasForwardField = ismember(deformationFieldDirection, {'forward', 'both', 'all'});
            hasBackwardField = ismember(deformationFieldDirection, {'backward', 'both', 'all'});
            if hasForwardField
                filesDeformationField{end+1,1} = tapas_uniqc_prefix_files(fileRaw, 'y_');
                filesDeformationField{end,1} = tapas_uniqc_prefix_files(filesDeformationField{end,1}, splitSuffix, 1);
                filesDeformationFieldProcessed{end+1,1} = ...
                    fullfile(pathSave, 'forwardDeformationField.nii');
            end
            if hasBackwardField
                filesDeformationField{end+1, 1} = tapas_uniqc_prefix_files(fileRaw, 'iy_');
                filesDeformationField{end,1} = tapas_uniqc_prefix_files(filesDeformationField{end,1}, splitSuffix, 1);
                filesDeformationFieldProcessed{end+1, 1} = ...
                    fullfile(pathSave, 'backwardDeformationField.nii');
            end
            
            % bias field names
            fileBiasField = cellstr(tapas_uniqc_prefix_files(fileRaw, ...
                'BiasField_'));
            fileBiasFieldProcessed = cellstr(fullfile(pathSave, ...
                'biasField.nii'));
            
            % join bias field and bias field corrected outputs
            % add wildcard for multi-channel segmentation
            biasFieldPath = dir(tapas_uniqc_prefix_files(fileBiasField{1}, '*', 1));
            if ~isempty(biasFieldPath)
                tmpBiasField = MrImage(tapas_uniqc_prefix_files(fileBiasField{1}, '*', 1), ...
                    'updateProperties', 'save');
                delete(tapas_uniqc_prefix_files(fileBiasField{1}, '*', 1));
                tmpBiasField.save();
            else
                this.save('fileName', fullfile(fileBiasField{1}));
            end
            
            % move all image files to their final names
            filesMoveSource = [
                filesTpm
                filesDeformationField
                fileBiasField
                ];
            filesMoveTarget = [
                filesTpmProcessed
                filesDeformationFieldProcessed
                fileBiasFieldProcessed
                ];
            tapas_uniqc_move_with_hdr(filesMoveSource, filesMoveTarget);
            
            % now load all output variables
            
            % load tissue probability maps
            doLoadTissueProbabilityMaps = nargin >= 1;
            if doLoadTissueProbabilityMaps
                nTissues = numel(filesTpmProcessed);
                for iTissue = 1:nTissues
                    varargout{1}{iTissue,1} = MrImage(filesTpmProcessed{iTissue}, ...
                        'updateProperties', 'all');
                end
            end
            
            % load deformation fields, if wanted
            doLoadDeformationFields = nargout >= 2;
            if doLoadDeformationFields
                varargout{2} = {};
                if hasForwardField
                    varargout{2}{1,1} = MrImage(...
                        filesDeformationFieldProcessed{1}, ...
                        'updateProperties', 'all');
                    % 5th dimension is the deformation dimensions
                    allDimLabels = varargout{2}{1,1}.dimInfo.dimLabels;
                    varargout{2}{1,1}.dimInfo.dimLabels{5} = ['d_', ...
                        allDimLabels{1}, allDimLabels{2}, allDimLabels{3}];
                end
                
                if hasBackwardField
                    varargout{2}{end+1,1} = MrImage(...
                        filesDeformationFieldProcessed{end}, ...
                        'updateProperties', 'all');
                    % 5th dimension is the deformation dimensions
                    allDimLabels = varargout{2}{end,1}.dimInfo.dimLabels;
                    varargout{2}{end,1}.dimInfo.dimLabels{5} = ['d_', ...
                        allDimLabels{1}, allDimLabels{2}, allDimLabels{3}];
                end
                
            end
            
            
            doLoadBiasField = nargout >= 3;
            if doLoadBiasField
                varargout{3}{1} = MrImage(fileBiasFieldProcessed, ...
                    'updateProperties', 'all');
            else
                varargout{3}{1} = {};
            end
            
            % other file with normalization information for old
            % segmentation
            fileSeg8 = regexprep(fileRaw, '\.nii$', '_seg8\.mat');
            
            % files to be deleted, if specified
            if doSaveProcessed
                filesCreated = [
                    filesCreated
                    fileSeg8
                    ];
            else
                filesCreated = [
                    filesCreated
                    filesMoveTarget
                    fileSeg8
                    ];
            end
            
        case 'smooth'
            prefixOutput = 's';
            fileOutputSpm = tapas_uniqc_prefix_files(fileRaw, prefixOutput);
    end
    
    % copy dimInfo to SPM-output file, if it exists
    % coregister has already written new file incl. dimInfo
    % segment does not change the image
    if ~any(strcmp(module, {'coregister_to'}))
        fileDimInfoRaw = this.get_filename('prefix', 'dimInfoRaw');
        if exist(fileDimInfoRaw, 'file')
            copyfile(fileDimInfoRaw, tapas_uniqc_prefix_files(fileDimInfoRaw, prefixOutput))
        end
    end
    
    tapas_uniqc_move_with_hdr(fileOutputSpm, fileProcessed);
    
    % load back data into matrix
    this.load(fileProcessed);
    
    % check whether dimInfo is available as well
    [pathstr,name] = fileparts(fileProcessed);
    dimInfoFileName = dir(fullfile(pathstr, [name, '_dimInfo.mat']));
    if ~isempty(dimInfoFileName) && ~isempty(dimInfoFileName.name)
        loadDimInfo = load(fullfile(dimInfoFileName.folder, dimInfoFileName.name));
        % add only non-geometry related properties, as they are already set
        % from the loaded file (and change e.g. for reslice)
        newDimInfo = MrDimInfo(loadDimInfo.objectAsStruct);
        nonGeomDims = find(~ismember(newDimInfo.dimLabels, {'x', 'y', 'z'}));
        for nUpdateDims = nonGeomDims
            dimLabel = newDimInfo.dimLabels{nUpdateDims};
            % dim exists but needs to be set to correct value
            if ismember(dimLabel, this.dimInfo.dimLabels)
                this.dimInfo.set_dims(dimLabel, ...
                    'units', newDimInfo.(dimLabel).units, ...
                    'samplingPoints', newDimInfo.(dimLabel).samplingPoints, ...
                    'samplingWidths', newDimInfo.(dimLabel).samplingWidths);
                % dim does not exist and needs to be added
            else
                this.dimInfo.add_dims(dimLabel, ...
                    'units', newDimInfo.(dimLabel).units{:}, ...
                    'samplingPoints', newDimInfo.(dimLabel).samplingPoints{:}, ...
                    'samplingWidths', newDimInfo.(dimLabel).samplingWidths);
            end
        end
        % also add dimInfo to the bias field
        if any(strcmp(module, {'segment'})) && ~isempty(biasFieldPath)
            varargout{3}{1}.dimInfo = newDimInfo.copyobj();
        end
    end
    
    % remove NaNs
    this.data(isnan(this.data)) = 0;
    
    % delete all unwanted files
    if ~doSaveRaw
        tapas_uniqc_delete_with_hdr(filesCreated);
        
        % delete raw sub-folder of current processing step
        if exist(pathRaw, 'dir')
            delete(fullfile(pathRaw, '*'));
            rmdir(pathRaw);
        end
        
        [stat, mess, id] = rmdir(this.parameters.save.path);
    end
else % no matlabbatch created
    
    % NOTE: this saving is important, e.g. for compute_masks to allow for
    % multiple processing steps within a bigger step to be saved
    if doSave
        this.save();
    end
end

% to preserve name after reloading, because not in nifti-file
this.name = nameImage;

end