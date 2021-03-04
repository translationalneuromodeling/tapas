classdef MrAffineTransformation < MrCopyData
    % Stores affine transformation for an image. Is disregarded during
    % display.
    % The order of the transformations follows the SPM convention: T*R*Z*S
    %   with T = translation, R = rotation, Z = Zoom (scaling), S = Shear
    %
    % Assumes that matrix always refers to dimensions in order
    % {'x', 'y', 'z'} => if dims are in different order in dimInfo, they
    % are resorted before applying a transformation.
    %
    % If only a file is given, the affine transformation from the header is
    % read.
    %
    % If created from a file and a dimInfo, it is assumed that the affine
    % transformation defined in dimInfo has to be removed from the affine
    % matrix stored in affineTransformation, such that the combination of dimInfo
    % and affineTransformation in MrImageGeometry gives the original affine
    % transformation described in the file.
    %
    % NOTE: If you want to see rotations/offcenter etc. in a different
    % coordinate system, look at MrImageGeometry
    %
    % EXAMPLE
    %   MrAffineTransformation
    %
    %   See also tapas_uniqc_spm_matrix tapas_uniqc_spm_imatrix MrDimInfo
    %   MrImageGeometry MrImage
    %   MrImageGeometry.set_from_dimInfo_and_affineTrafo
    
    % Author:   Saskia Bollmann & Lars Kasper
    % Created:  2016-06-15
    % Copyright (C) 2016 Institute for Biomedical Engineering
    %                    University of Zurich and ETH Zurich
    
    % This file is part of the TAPAS UniQC Toolbox, which is released
    % under the terms of the GNU General Public License (GPL), version 3.
    % You can redistribute it and/or modify it under the terms of the GPL
    % (either version 3 or, at your option, any later version).
    % For further details, see the file COPYING or
    %  <http://www.gnu.org/licenses/>.
    
    properties
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
        
        % scaling/zoom factor, typically 1, unless resolution is
        % temporarily stored
        scaling         = [1 1 1];
        
    end % properties
    
    properties (Dependent)
        % Affine transformation matrix, computed from SPM
        % combines operations as T*R*Z*S
        affineMatrix;
    end
    
    methods
        
        function this = MrAffineTransformation(varargin)
            % Constructor of class
            %   MrAffineTransformation(affineMatrix)
            %       OR
            %   MrAffineTransformation(fileName)
            %       OR
            %   MrAffineTransformation(fileName, dimInfo)
            %       OR
            %   MrAffineTransformation(affineMatrix, dimInfo)
            %       OR
            %   MrAffineTransformation('PropertyName', PropertyValue, ...)
            
            hasInputFile = nargin == 1 && ischar(varargin{1}) && exist(varargin{1}, 'file');
            hasInputAffineMatrix = nargin == 1 && isnumeric(varargin{1});
            hasInputFileAndDimInfo = nargin == 2 && ischar(varargin{1}) && exist(varargin{1}, 'file') ...
                && isa(varargin{2}, 'MrDimInfo');
            hasInputAffineMatrixAndDimInfo = nargin == 2 && isnumeric(varargin{1}) ...
                && isa(varargin{2}, 'MrDimInfo');
            
            if hasInputFile
                % load from file
                this.load(varargin{1});
            elseif hasInputAffineMatrix
                % affineMatrix
                this.update_from_affine_matrix(varargin{1});
            elseif hasInputFileAndDimInfo
                % load from file
                this.load(varargin{1});
                % get affine transformation from dimInfo
                ADimInfo = varargin{2}.get_affine_matrix;
                % update affineTransformation
                this.update_from_affine_matrix(this.affineMatrix/ADimInfo);
            elseif hasInputAffineMatrixAndDimInfo
                % update from affine matrix 
                this.update_from_affine_matrix(varargin{1});
                % get affine transformation from dimInfo
                ADimInfo = varargin{2}.get_affine_matrix;
                % update affineTransformation
                this.update_from_affine_matrix(this.affineMatrix/ADimInfo);
            else
                for cnt = 1:nargin/2 % save 'PropertyName', PropertyValue  ... to object properties
                    this.(varargin{2*cnt-1}) = varargin{2*cnt};
                end
            end
        end
        
        % NOTE: Most of the methods are saved in separate function.m-files in this folder;
        %       except: constructor, delete, set/get methods for properties.
        
        function affineMatrix = get.affineMatrix(this)
            affineMatrix = this.get_affine_matrix();
        end
        
    end % methods
    
end
