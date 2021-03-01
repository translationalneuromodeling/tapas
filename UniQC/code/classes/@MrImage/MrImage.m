classdef MrImage < MrDataNd
    % An n-dimensional MR image, on which typical image algebra operations 
    % can be performed (similar, e.g. to fslmaths), but on top, image
    % processing operations (e.g. erosion/dilation) of the image processing
    % toolbox. Furthermore, interfaces with neuroimaging formats from
    % different vendors (Philips) and nifti is supported.
    %
    %   Y = MrImage(dataMatrix, 'propertyName', propertyValue, ...)
    %       OR
    %   Y = MrImage(fileName, 'propertyName', propertyValue, ...)
    %
    % IN
    %   dataMatrix  n-dimensional matrix with the following dimension order
    %
    %               OR
    %
    %   figureHandle/axesHandle 
    %               creates image from CData within current Image of
    %               specified figure/axis
    %
    %               OR
    %   folderName 
    %               with files of types below
    %           
    %   fileName    string or cell of strings; if cell is given, image files
    %               have to have the same 3D geometry and are appended to
    %               an  n-dimensional MrImage
    %
    %              - supported file-types:
    %              .nii         nifti, header info used
    %              .img/.hdr    analyze, header info used
    %              .cpx         Philips native complex (and coilwise) image
    %                           data format
    %              .par/.rec    Philips native image file format
    %              .mat         matlab file, assumes data matrix in
    %                           variable 'data'
    %                           and parameters in 'parameters' (optional)
    %
    %   'PropertyName'/value - pairs possible:
    %               'imageType'         'abs' or 'angle'/'phase'
    %                                   default: 'abs'
    %                                   (only for par/rec data)
    %               'iEcho'             echo number to be loaded
    %                                   default: 1
    %                                   (only for par/rec data)
    %               'selectedCoils'     [1,nCoils] vector of selected Coils
    %                                   to be loaded (default: 1)
    %               'selectedVolumes'   [1,nVols] vector of selected volumes
    %                                   to be loaded
    %               'signalPart'        'abs'       - absolute value
    %                                   'phase'     - phase of signal
    %               'updateProperties'  (cell of) strings containing the
    %                                   properties of the object to be
    %                                   updated with the new (file)name and
    %                                   its data
    %                                       'name'  name is set to file name
    %                                              (default)
    %                                       'save'  parameters.save.path and
    %                                               parameters.save.fileName
    %                                               are updated to match
    %                                               the input file name
    %                                       'none'  only data and geometry
    %                                               updated by loading
    %                                       'all'   equivalent to
    %                                       {'name','save'}
    %
    %               properties of MrImageGeometry; See also MrImageGeometry
    %               e.g.
    %               'resolution_mm'    , [1 1 1]
    %               'offcenter_mm'     , [0 0 0]
    %               'rotation_deg'     , [0 0 0]
    %               'shear'         , [0 0 0]
    %
    %
    % OUT
    %
    % EXAMPLE
    %   Y = MrImage(dataMatrix, 'resolution_mm', [2.5 2.5 4], ...
    %       'FOV_mm', [220 220 110], 'TR_s', 3)
    %   Y = MrImage('spm12b/canonical/single_subj_T1.nii')
    %   Y = MrImage(gcf)
    %
    %   See also MrImage.load MrDimInfo MrImageGeometry MrDataNd MrAffineTransformation
    
    % Author:   Saskia Klein & Lars Kasper
    % Created:  2014-04-15
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
        
        % other properties: See also MrDataNd
        rois    = []; % see also MrRoi
        
        % TODO: add the acquisition parameters? useful for 'advanced' image
        % processing such as unwrapping and B0 computation.
        
        affineTransformation = [] % MrAffineTransformation
    end
    
    properties (Dependent = true)
        % geometry of a slab is both the extent of the slab (FOV, resolution, nVoxels
        %   => dimInfo
        % and its position and orientation in space (affineTransformation)
        % geometry is thus a dependent property (no set (?)) formed as a
        % combination of the two.
        % See also MrImageGeometry
        %
        % 3D Geometry properties of data-matrix + 4D time info,
        % in particular for save/load from nifti/par-rec for fMRI
        % provides full voxel to world mapping, i.e. affine transformation
        % including rotation/translation/voxel scaling
        geometry
    end
    
    methods
        
        % NOTE: Most of the methods are saved in separate function.m-files in this folder;
        %       except: constructor, delete, set/get methods for properties.
            
        function this = MrImage(varargin)
            % Constructor of MrImage class. Accepts fileName input for different
            % file type (nifti, analyze, mat):
            % EXAMPLES
            % Y = MrImage('filename.nii')
            %       nifti files, header is read to update MrImage.parameters
            % Y = MrImage('filename.img') or Y = MrImage('filename.hdr')
            %       analyze files, header is read to update MrImage.parameters
            % Y = MrImage('filename.mat', 'PropertyName', PropertyValue, ...)
            %       matlab matrix loaded from file, specify
            %       properties:
            %           dimInfo     MrDimInfo  e.g. resolutions, dimLabels
            %                                  ranges, ...)
            % Y = MrImage(variableName, 'PropertyName', PropertyValue, ...)
            %       matlab matrix "variableName" loaded from workspace
            % Y = MrImage(gcf);
            %       2D image created from line or image plots in current figure
            % Y = MrImage(gca);
            %       2D image created from line or image plots in current
            %       axes
            % Y = MrImage(figure(121)); 
            %       2D image created from line or image plots in figure 121
            %       figure call is needed to distinguish figure handle from 
            %       single number image with value 121
            %
            % uses MrDataNd.load
            this@MrDataNd(varargin{:});
            
            % initialize, if not read in by MrDataNd constructor
            if isempty(this.affineTransformation)
                this.affineTransformation = MrAffineTransformation();
            end
            
            this.parameters.save.path = regexprep(this.parameters.save.path, 'MrDataNd', class(this));
            % only overwrite if default (MrDataNd) is set
            if strcmp(this.parameters.save.fileName, 'MrDataNd.mat')
                this.parameters.save.fileName = 'MrImage.nii';
            end
            if strcmp(this.name, 'MrDataNd')
                this.name = 'MrImage';
            end
            % Call SPM job manager initialisation, if not done already.
            % Check via certain matlabbatch-function being on path
            if ~exist('cfg_files', 'file')
                if exist('spm_jobman', 'file')
                    pathSpm = fileparts(which('spm'));
                    % remove subfolders of SPM, since it is recommended, 
                    % and fieldtrip creates conflicts with Matlab functions otherwise
                    % check whether fieldtrip is on path
                    fieldtripPath = which('fieldtrip2fiff.m');
                    if ~isempty(fieldtripPath)
                        rmpath(genpath(pathSpm));
                        addpath(pathSpm);
                    end
                    spm_jobman('initcfg');
                elseif ~strcmp(getenv('BIOTOPE'),'EULER_Matthias') % don't want that
                    warning(sprintf(['SPM (Statistical Parametric Mapping) Software not found.\n', ...
                        'Some fMRI-related functionality will not work:\n', ...
                        '- See methods of MrImageSpm4D in folder (@MrImageSpm4D) \n', ...
                        '- These are used in MrSeries operations (realign/coregister/GLM) \n\n', ...
                        'For complete utility, Please add SPM (without its subfolders) to Matlab path or install from http://www.fil.ion.ucl.ac.uk/spm/']));
                end
            end
            
        end
        
        function geometry = get.geometry(this)
            % Get-Method for geometry
            % NOTE: no set method exists, since this is generated in
            % real-time from current dimInfo and affineTransformation
            %
            % geometry of a slab is both the extent of the slab (FOV, resolution, nVoxels
            %   => dimInfo
            % and its position and orientation in space (affineTransformation)
            % geometry is thus a dependent property set formed as a
            % combination of the two.
            % See also MrImageGeometry
            
            geometry = MrImageGeometry(this.dimInfo, this.affineTransformation);
            
            props = properties(geometry);
            
            for p = 1:numel(props)
                addlistener(geometry, props{p}, 'PreSet', @(p,h) set_geometry_callback(this, p, h));
            end
        end
        
    end
end