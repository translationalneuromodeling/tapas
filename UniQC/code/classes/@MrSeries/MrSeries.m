classdef MrSeries < MrCopyData
    % Class of MR Time Series analysis(4D = spatial coordinates:x,y,z, and time)
    % 
    % TODO: Revert all MrImageSpm4D properties to MrImage, once methods are made
    %       nD-compatible
    %
    % EXAMPLE
    %   MrSeries
    %
    %   See also
    
    % Author:   Saskia Klein & Lars Kasper
    % Created:  2014-06-06
    % Copyright (C) 2014 Institute for Biomedical Engineering
    %                    University of Zurich and ETH Zurich
    %
    % This file is part of the TAPAS UniQC Toolbox, which is released
    % under the terms of the GNU General Public Licence (GPL), version 3.
    % You can redistribute it and/or modify it under the terms of the GPL
    % (either version 3 or, at your option, any later version).
    % For further details, see the file COPYING or
    %  <http://www.gnu.org/licenses/>.
    %

    
    properties
        name    = 'MrSeries'; % String identifier of MrSeries-object
                
        data    = []; % MrImageSpm4D(); % contains nX*nY*nZ*nT data matrix (also called data)
        
        mean    = []; % MrImageSpm4D(); % mean image over volumes of time series
        sd      = []; % MrImageSpm4D(); % standard deviation image over volumes of time series
        snr     = []; % MrImageSpm4D(); % signal-to-noise ratio (snr) image over volumes of time series
        coeffVar = []; % MrImageSpm4D(); % coefficient of variation
        % difference image between first and last volume of time series
        % characterizing drift
        diffLastFirst = []; % MrImageSpm4D();
        % difference image between odd and even volumes of time series,
        % characterizing image (not temporal) noise (Friedman & Glover,
        % JMRI 2006)
        diffOddEven = []; % MrImageSpm4D();
        
        anatomy = []; % MrImageSpm4D();  % anatomical image for reference
        tissueProbabilityMaps = {} % cell of MrImages, tissue probability maps
        masks   = {}; % cell of MrImages
        rois    = {}; % cell of MrRois
       
        % General linear model
        glm     = []; % MrGlm();
        
        % parameters for all complicated methods
        parameters = ...
            struct(...
            'compute_stat_images', ...
            struct( ...
            'applicationDimension', 't' ...
            ), ...
             'compute_tissue_probability_maps', ...
            struct( ...
            'nameInputImage', 'mean', ...
            'tissueTypes', {{'GM', 'WM', 'CSF'}} ...
            ), ...
             'compute_masks', ...
            struct( ...
            'nameInputImages', 'tissueProbabilityMap', ...        % String with image name (or search pattern) from which masks shall be created
            'threshold', 0.9, ...               % Threshold at and above mask shall be equal 1
            'keepExistingMasks', true, ...      % If true, existing Images in masks-cell are retained, new masks appended; if false, masks is overwritten by new masks
            'nameTargetGeometry', 'data' ...    % String with image name (or search pattern) to which masks shall be resliced
            ), ...
            'coregister', ...
            struct( ...
               'nameStationaryImage', [], ...
               'nameTransformedImage', [], ...
               'nameEquallyTransformedImages', [], ...
               'affineCoregistrationGeometry', [] ...
            ), ...
            'analyze_rois', ...
            struct( ...
            'nameInputImages', {{'snr', 'sd'}}, ...
            'nameInputMasks', 'mask', ...
            'keepCreatedRois', true ...
            ), ...
            'realign', ...
            struct( ...
            'quality', 0.9 ...
            ), ...
            'smooth', struct('fwhmMillimeters', 8), ...
            't_filter', ...
            struct( ...
            'cutoffSeconds', 128 ...
            ), ...
            'save', ...
            struct( ...
            'path', '', ...
            'format', 'nii', ...
            'items', 'processed' ... % items to save: 'none', 'all', 'object', 'processed' % 'all' also keeps raw files
            ) ...
            );

            % cell of MrImages that might be used for mask-creation or similar
            % e.g. deformation fields, bias fields, anatomical
            % atlases/masks
            additionalImages = {}; 
            
        % Cell(nProcessingSteps,1) of numbered processing steps performed 
        % on the MrSeries since its creation
        processingLog = {};         
        svnVersion = '$Rev$'; % code version
        nProcessingSteps = 0;
        end % properties
    
    
    methods
        
        function this = MrSeries(fileName, varargin)
        % Constructor of class, can be initialized as MrImageSpm4D with 
        %
        %   this = MrSeries(fileName, varargin)
        %
        % IN
        %   fileName    an image file (or data matrix) holding the image
        %               time series to be analyzed (with SPM)
        %   varargin    propertyName/Value pairs, setting specific
        %               properties of the MrImageSpm4D used as image series
        %
            if ~exist('cfg_files', 'file')
                if exist('spm_jobman')
                    spm_jobman('initcfg');
                else
                    warning(sprintf(['SPM (Statistical Parametric Mapping) Software not found.\n', ...
                        'Some fMRI-related functionality, esp. of MrSeries, will not work. \n\n', ...
                        'For complete utility, Please add SPM to Matlab path or install from http://www.fil.ion.ucl.ac.uk/spm/']));
                end
            end
            
            % object initialization before value are altered
            %this = this@MrCopyData();
            
            % construct all objects that are properties of MrSeries within
            % this constructor to to avoid weird pointer Matlab bug
            % that set default values to first created object
            
            this.glm = MrGlm();
            this.parameters.coregister.affineCoregistrationGeometry = ...
                MrImageGeometry();
            
            % create MrImages in constructor
            % also: set default names for statistical images as properties
            [~, nameImageArray] = this.get_all_image_objects();
            for iImage = 1:numel(nameImageArray)
                this.(nameImageArray{iImage}) = MrImageSpm4D(); % TODO: Revert to MrImage, once all methods migrated for nD-data
                img =  this.(nameImageArray{iImage});
                img.name = nameImageArray{iImage};
                img.parameters.save.fileName = ...
                    [nameImageArray{iImage} '.nii'];
            end
            
            % save path
            stringTime = datestr(now, 'yymmdd_HHMMSS');
            pathSave = fullfile(pwd, ['MrSeries_' stringTime]);
            this.set_save_path(pathSave);
          
            
            switch nargin
                case 0
                    
                otherwise
                    %somehow, all variable parameters are converted
                    %into a cell, if varargin is given directly...
                    this.load(fileName,varargin{:});
            end
        end
        
    end % methods
    
end
