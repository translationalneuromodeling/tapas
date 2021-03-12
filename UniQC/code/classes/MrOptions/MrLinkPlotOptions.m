classdef MrLinkPlotOptions
% This is a predefined options/parameter structure for MrImage.plot('linkOptions', outputStruct)
%
%   outputStruct = MrLinkPlotOptions(input)
%
%
%   See also MrImage.plot
 
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-01-05
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

properties
    % string, type of plot to be linked, default: timeseries
    % possible values:
    %   'timeseries/ts'
    plotType = 'timeseries';
    % integer, dimension of MR Image which should be plotted; default: 4
    plotDim = 4;
    % cell(1,2*nFixedDims)  dimension label/index array pairs thatshall remain
    %                       fixed/selected and are not on original plot
    %                           default: {5 ,1, 6, 1, ..., nDims, 1}
    %                           or:      {dimLabel5, 1, dimLabel6, 1, ..., dimLabelN, 1}
    fixedDimsPoint = {};
    % function handle (or string) to convert current mouse position in
    % proper position selection (x,y,z)
    % default:  swapping first and second mouse coordinate to reflext
    %           displays/matrix order dim order difference in Matlab
    convertMousePosToSelection = @(x) [x(2) x(1) 1];
end
methods
    function this = MrLinkPlotOptions(varargin)
        % Constructor of class MrLinkPlotOptions, allows 'PropertyName', PropertyValue pairs
        % everything else set to default ([])
        % Further valid constructor calls
        %   MrLinkPlotOptions('timeseries', dimInfo, parentPlotDims, plotDim)
        %   MrLinkPlotOptions('ts', dimInfo, parentPlotDims, plotDim)
        %
        %   IN
        %       dimInfo - of MrImage that is plotted
        %       parentPlotDims
        %               - dimensions that are plotted in original plot
        %       plotDim - dimension used for timeseries plot; default: 4
        if nargin
            if ~ismember(varargin{1}, properties(MrLinkPlotOptions)) 
            % shortcut strings
                switch varargin{1}
                    case {'timeseries', 'ts'}
                        dimInfo = varargin{2};
                        parentPlotDims = varargin{3};
                        plotDim = 4;
                        if nargin >= 4
                            plotDim = varargin{4};
                        end
                        
                        % find all dims that are neither plotted in this
                        % nor the linked plot
                        idxFixedDims = dimInfo.get_dim_index(parentPlotDims, ...
                            'invert', true);
                        idxFixedDims = setdiff(idxFixedDims, plotDim);
                        
                        % assemble fixedDims Point by two column cell array
                        % with selected label dims and first sampling point
                        fixedDimLabels = dimInfo.dimLabels(...
                            idxFixedDims)';
                        fixedDimValues = ...
                            cellfun(@(x) {x(1)}, dimInfo.samplingPoints(idxFixedDims))';
                        this.fixedDimsPoint = reshape([fixedDimLabels fixedDimValues], 1, []);
                    otherwise
                        error('tapas:uniqc:MrLinkPlotOptions:UnknownShortcutConstructor', ...
                            'Unknown shortcut constructor for class MrLinkPlotOptions');
                end
            else
            % propname/value pairs
                for cnt = 1:nargin/2 % save them to object properties
                this.(varargin{2*cnt-1}) = varargin{2*cnt};
                end
            end    
        end
    end
end
end
