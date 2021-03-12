classdef MrCompare < MrCopyData
    % This class loads and compares different objects, e.g. images/rois
    %
    % EXAMPLE
    %   MrCompare(imageArray, extraDimInfo)
    %
    %   See also MrDimInfo MrImage
    
    % Author:   Saskia Klein & Lars Kasper
    % Created:  2019-02-25
    % Copyright (C) 2019 Institute for Biomedical Engineering
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
        
        % cell(nRows,1) of strings with detailed image information, e.g.
        % previous processing steps to arrive at that image
        % (masking/thresholding...)
        % 'detailed image information'; 'given in cell strings'
        info    = {};
        
        % A short string identifier of the image, used e.g. as plot-title
        name    = 'MrCompare';
        
        % cell(nExtraDim1, ..., nExtraDimN) holding handles to all objects
        % that shall be compared in different aspects
        % The dimensions are described by the dimInfo
        data = {};
        
        % MrDimInfo for data, describing which items are held in data
        dimInfo
        
        
    end % properties
    
    
    methods
        
        function this = MrCompare(objectArray, extraDimInfo)
            % Constructor of class
            % IN
            %   objectArray cell(nExtraDim1, ..., nExtraDimN)
            %                   OR
            %               cell(nObjects,1)
            %               holding handles to all objects of the same class
            %               (MrImage, MrRoi, MrSeries, or MrDataNd,
            %               MrImageSpm4D) to be included in comparison
            %   extraDimInfo
            %               MrDimInfo, describing meta-data of objectArray,
            %               e.g., objects from different subjects, sessions,
            %               protocols
            
            
            if nargin >= 1 % needed to allow empty constructor for copyobj etc
                % select only non-singleton-dims for dimInfo
                nSamples = size(objectArray);
                iDims = nSamples > 1;
                nDims = max(sum(iDims),1);
                nSamples = nSamples(iDims);
                if isempty(nSamples), nSamples = 1; end;
                
                
                if nargin >= 2
                    this.dimInfo = extraDimInfo.copyobj();
                else
                    
                    
                    for iDim = 1:nDims
                        dimLabels{iDim} = sprintf('dim%d', iDim);
                        units{iDim}      = '';
                    end
                    this.dimInfo = MrDimInfo('dimLabels', dimLabels, 'units', units, ...
                        'nSamples', nSamples);
                end
                
                % needed for reshape...
                if nDims == 1
                    nSamples(2) = 1;
                end
                
                this.data = reshape(squeeze(objectArray),nSamples);
                
                for iObject = 1:numel(this.data);
                    this.name = sprintf('%s %s', this.name, objectArray{iObject}.name);
                end
                this.info{end+1,1} = sprintf('Constructed from %s', this.name);
            end
        end
        
        % NOTE: Most of the methods are saved in separate function.m-files in this folder;
        %       except: constructor, delete, set/get methods for properties.
        
    end % methods
    
end