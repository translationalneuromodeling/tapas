classdef MrDataNd < MrCopyData
    % Class for n-dimensional data representation with dimensional information,
    % selection, plotting and general unary and binary arithmetic operations
    %
    %
    % EXAMPLE
    %   MrDataNd('test.nii');
    %   MrDataNd(dataMatrix, 'dimInfo', myDimInfo);
    %
    %   See also MrImage MrDimInfo
    
    % Author:   Saskia Klein & Lars Kasper
    % Created:  2016-04-03
    % Copyright (C) 2016 Institute for Biomedical Engineering
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
        % Parameter Structure about image data storage
        parameters = struct( ...
            'save', struct( ...
            'path', './zFatTmp', ...  % path where disk files can be stored temporarily
            'fileName', 'MrDataNd.mat', ... %  file name for saving
            'keepCreatedFiles', 'none' ... % 'none', 'all', 'processed' keep temporary files on disk after method finished
            ), ...
            'plot', struct('dimOrder', [1 2 3]), ...
            'verbose', struct('level', 0) ...
            );
        
        % cell(nRows,1) of strings with detailed image information, e.g.
        % previous processing steps to arrive at that image
        % (masking/thresholding...)
        % 'detailed image information'; 'given in cell strings'
        info    = {};
        
        % A short string identifier of the image, used e.g. as plot-title
        name    = 'MrDataNd';
        
        data    = []; % nX*nY*nZ*nVolumes...etc N-Dimensional data matrix
        
        % Dimensionality information about image (names of dimensions,
        % units, array size, sampling-points, also for n-dimensional data,
        % e.g. echo times, coil indices)
        dimInfo  = []; % see also MrDimInfo
        
    end % properties
    
    
    methods
        
        function this = MrDataNd(inputDataOrFile, varargin)
            % Constructor of class
            %   Y = MrDataNd(fileName/folderName, 'propertyName', propertyValue, ...);
            % OR
            %   Y = MrDataNd(figureHandle/axisHandle, 'propertyName', propertyValue, ...);
            % OR
            %   Y = MrDataNd(folderName, 'propertyName', propertyValue, ...);
            % OR
            %   Y = MrDataNd(dataMatrix, 'propertyName', propertyValue, ...);
            % OR
            %   Y = MrDataNd(fileNameCellArray, 'propertyName', propertyValue, ...);
            % OR
            %   Y = MrDataNd(fileNameSearchString, 'propertyName', propertyValue, ...);
            %
            % transfer all properties given as name/value pairs to object
            % call superclass constructor
            this@MrCopyData(varargin{:});
            
            defaults.dimInfo = MrDimInfo();
            defaults.name = 'MrDataNd';
            defaults.info = {};
            
            [args, argsUnused] = tapas_uniqc_propval(varargin, defaults);
            
            % populate object with input data prop/value pairs
            props = fieldnames(defaults);
            for p = 1:numel(props)
                this.(props{p}) = args.(props{p});
            end
            
            % save path
            stringTime = datestr(now, 'yymmdd_HHMMSS');
            pathSave = fullfile(pwd, ['MrDataNd_' stringTime]);
            this.parameters.save.path = pathSave;
            
            % load data, and update dimInfo
            if nargin >=1
                this.load(inputDataOrFile, varargin{:});
            end
            
        end
        
        % NOTE: Most of the methods are saved in separate function.m-files in this folder;
        %       except: constructor, delete, set/get methods for properties.
        
    end % methods
    
end
