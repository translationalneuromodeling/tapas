classdef MrImageGeometry < MrCopyData
    % Geometry properties of MrImage, in particular for operations on niftis
    % Provides full voxel to world mapping, i.e. affine transformation
    % Including rotation/translation/voxel scaling
    %
    %
    % EXAMPLE
    %   MrImageGeometry
    %
    %   See also MrImage tapas_uniqc_spm_matrix tapas_uniqc_spm_imatrix
    
    % Author:   Saskia Bollmann & Lars Kasper
    % Created:  2014-07-15
    % Copyright (C) 2014 Institute for Biomedical Engineering
    %                    University of Zurich and ETH Zurich
    
    % This file is part of the TAPAS UniQC Toolbox, which is released
    % under the terms of the GNU General Public Licence (GPL), version 3.
    % You can redistribute it and/or modify it under the terms of the GPL
    % (either version 3 or, at your option, any later version).
    % For further details, see the file COPYING or
    %  <http://www.gnu.org/licenses/>.
    
    properties (SetObservable = true)
        
        % [1,3] vector of Field of View (in mm)
        FOV_mm          = [0 0 0];
        
        % [1,4] vector of number of voxels per image dimension
        % (x, y, z and time (number of volumes)
        nVoxels         = [1 1 1 1];
        
        % [1,3] vector of image resolution (voxel size in mm) in x,y,z
        % direction
        resolution_mm   = [0 0 0];
        
        % Repetition time in seconds
        % between subsequent scans/volumes (4th dim samples)
        TR_s            = 0;
        
        % [1,3] vector of translational offcenter (in mm) in x,y,z of
        % image volume with respect to isocenter
        offcenter_mm    = [0 0 0];
        
        % [1,3] vector of rotation (in degrees)
        % around x,y,z-axis (i.e. pitch, roll and yaw), i.e. isocenter (0,0,0)
        rotation_deg    = [0 0 0];
        
        % [1,3] vector of y->x, z->x and z->y shear factor of coordinate
        %
        % equivalent to off-diagonal elements of affine transformation matrix:
        % S   = [1      P(10)   P(11)   0;
        %        0      1       P(12)   0;
        %        0      0       1       0;
        %        0      0       0       1];
        shear           = [0 0 0];
        
    end % properties
    
    methods
        
        function this = MrImageGeometry(varargin)
            % Constructor of class, allows input of affine transformation matrix or
            % nifti/analyze file parsing its header information
            %
            %   MrImageGeometry(fileName, 'PropertyName', PropertyValue, ...)
            %   MrImageGeometry([], 'PropertyName', PropertyValue, ...)
            %   MrImageGeometry(dimInfo, affineTransformation)
            %   MrImageGeometry(affineMatrix)
            
            hasInputFile = nargin && ~isempty(varargin{1}) ...
                && ischar(varargin{1});
            hasInputMatrix = nargin && ~isempty(varargin{1}) ...
                && isa(varargin{1}, 'numeric');
            hasOneValidInput = nargin && ~isempty(varargin{1});
            hasTwoValidInputs = nargin > 1 && ~isempty(varargin{2});
            
            % check whether dimInfo is first input
            isDimInfoFirstInput = hasOneValidInput && isa(varargin{1}, 'MrDimInfo');
            isAffineTransformationSecondInput = hasTwoValidInputs && isa(varargin{2}, 'MrAffineTransformation');
            isAffineTransformationFirstInput = hasOneValidInput && isa(varargin{1}, 'MrAffineTransformation');
            isDimInfoSecondInput = hasTwoValidInputs && isa(varargin{2}, 'MrDimInfo');
            hasInputObjects = (isDimInfoFirstInput  && isAffineTransformationSecondInput) ...
                || (isAffineTransformationFirstInput && isDimInfoSecondInput);
            
            if hasInputFile % file is provided
                fileName = varargin{1};
                tempDimInfo = MrDimInfo(fileName);
                % check whether individual file or whole folder is provided
                if isdir(fileName)
                    % if whole folder, read first file
                    tempDir = dir(fileName);
                    tempaffineTransformation = MrAffineTransformation(...
                        fullfile(fileName, tempDir(3).name), tempDimInfo);
                else
                    tempaffineTransformation = ...
                        MrAffineTransformation(fileName, tempDimInfo);
                end
                this.set_from_dimInfo_and_affineTrafo(tempDimInfo, tempaffineTransformation);
                hasInputObjects = 0;
            elseif hasInputObjects % dimInfo and affineTransformation are provided
                if isDimInfoFirstInput
                    this.set_from_dimInfo_and_affineTrafo(varargin{1}, varargin{2});
                elseif isAffineTransformationFirstInput
                    this.set_from_dimInfo_and_affineTrafo(varargin{2}, varargin{1});
                end
            elseif isDimInfoFirstInput && ~isAffineTransformationFirstInput
                % make empty affine transformation
                affineTransformation = MrAffineTransformation();
                this.set_from_dimInfo_and_affineTrafo(varargin{1}, affineTransformation);
            elseif isAffineTransformationFirstInput && ~isDimInfoSecondInput
                % make empty dimInfo
                dimInfo = MrDimInfo('firstSamplingPoint', [0 0 0], ...
                    'resolutions', [1 1 1], 'samplingWidths', [1 1 1]);
                this.set_from_dimInfo_and_affineTrafo(dimInfo, varargin{1});
            elseif hasInputMatrix
                % make empty dimInfo
                dimInfo = MrDimInfo('firstSamplingPoint', [0 0 0], ...
                    'resolutions', [1 1 1], 'samplingWidths', [1 1 1]);
                affineTransformation = MrAffineTransformation(varargin{1});
                this.set_from_dimInfo_and_affineTrafo(dimInfo, affineTransformation);
            end
            % update explicit geometry parameters
            % input file and additional parameters are given
            if hasInputFile && (nargin > 1)
                this.update(varargin{2:end});
                % input objects and additional parameters are given
            elseif hasInputObjects && (nargin > 2)
                this.update(varargin{3:end});
                % only additional parameters are given
            elseif ~hasInputFile && ~hasInputObjects && (nargin > 2)
                this.update(varargin{2:end});
            end
        end
        
        % NOTE: Most of the methods are saved in separate function.m-files in this folder;
        %       except: constructor, delete, set/get methods for properties.
    end % methods
    
end
