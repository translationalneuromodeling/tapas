classdef MrImageSpm4D < MrImage
    % 4D Image class that interfaces image processing operations of SPM
    %( e.g. realign, coreg, smooth, normalization, segmentation, temporal filter)
    %
    % NOTE:
    %   will convert into real-valued
    %
    % IN
    %   dataMatrix    [nVoxelX, nVoxelY, nSlices, nVolumes]
    %
    % OUT
    %
    % EXAMPLE
    %   MrImageSpm4D
    %
    %   See also
    
    % Author:   Saskia Bollmann & Lars Kasper
    % Created:  2018-01-11
    % Copyright (C) 2018 Institute for Biomedical Engineering
    %                    University of Zurich and ETH Zurich
    %
    % This file is part of the TAPAS UniQC Toolbox, which is released
    % under the terms of the GNU General Public License (GPL), version 3.
    % You can redistribute it and/or modify it under the terms of the GPL
    % (either version 3 or, at your option, any later version).
    % For further details, see the file COPYING or
    %  <http://www.gnu.org/licenses/>.
    %

    
    properties
        % has identical properties as MrImage, but enforced 4D image
    end % properties
    
    
    methods
        
        
        function this = MrImageSpm4D(varargin)
            % Constructor of class MrImageSpm4D
            % See also MrImage
            %
            % USAGE
            % 1) as MrImage-constructor
            
            this@MrImage(varargin{:});
            
        end
        
        % NOTE: Most of the methods are saved in separate function.m-files in this folder;
        %       except: constructor, delete, set/get methods for properties.
        
    end % methods
    
end
