classdef MrRoi < MrCopyData
    %class for regions of interest of an MrImage OR MrSeries
    %
    %
    % EXAMPLE
    %   MrRoi
    %
    %   See also MrImage MrSeries
    
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
    %

    
    properties
        
        data       % {nSlices,1} cell of [nVoxelSli,nScans] matrices for 4D data
        %  {nSlices,1} cell of [nVoxelSli,1] voxel values
        name       % string (e.g. name of mask and data input combined)
        nSlices = [];  % number of slices in original mask (also, if empty)
        nVolumes = []; % number of volumes Roi is extracted from
        
        parameters = struct( ...
            'save', struct( ...
            'path', './zFatTmp', ...  % path where disk files can be stored temporarily
            'fileName', 'roiData.nii', ... %  file name for saving
            'keepCreatedFiles', 'none' ... % 'none', 'all', 'processed' keep temporary files on disk after method finished
            ) ...
            );
        
        perSlice = struct( ...
            'mean', [], ...
            'sd', [], ...
            'snr', [], ...
            'coeffVar', [], ...
            'diffLastFirst', [], ...
            'min', [], ...
            'max', [], ...
            'median', [], ...
            'nVoxels', [] ...
            );
        perVolume = struct( ...
            'mean', [], ...
            'sd', [], ...
            'snr', [], ...
            'coeffVar', [], ...
            'diffLastFirst', [], ...
            'min', [], ...
            'max', [], ...
            'median', [], ...
            'nVoxels', [] ...
            );
    end % properties
    
    
    methods
        
        function this = MrRoi(image, mask)
            % Constructor of class, extra
            %   IN
            %       image   MrImage of 3D or 4D data
            %       mask    MrImage (3D) of binary values (1 = within mask; 0 = not in mask)
            
            if nargin
                this.extract(image,mask);
            end
        end
        
        % NOTE: Most of the methods are saved in separate function.m-files in this folder;
        %       except: constructor, delete, set/get methods for properties.
    end
end
