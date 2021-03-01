function this = MrDimInfo_select(this, testVariantsDimInfoSelect)
% Unit test for MrDimInfo select.
%
%   Y = MrUnitTest()
%   run(Y, 'MrDimInfo_select')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDimInfo_select
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2018-02-15
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% construct MrDimInfo object from sampling points
dimInfo = this.make_dimInfo_reference(0);

switch testVariantsDimInfoSelect
    case 'singleDim'
        selDim = 3;
        selDimChar = dimInfo.dimLabels{selDim};
        selArray = [10,13,16];
        % test if constant sampling interval is specified here (since we
        % want to have resolutions)
        indexDistance = diff(selArray);
        uniqueDistance = unique(indexDistance);
        isConstantSampling = numel(uniqueDistance) == 1;
        this.assertTrue(isConstantSampling, ...
            'Selection Array with uneuqal distance specified');
        % select DimInfo
        selectionDimInfo = dimInfo.select(selDimChar, selArray);
        
        % make both dimInfos to struct to make sure we are not calling
        % select again
        warning('off', 'MATLAB:structOnObject');
        dimInfoStruct = struct(dimInfo);
        selectionDimInfoStruct = struct(selectionDimInfo);
        warning('on', 'MATLAB:structOnObject');
        
        % select in struct
        dimInfoStruct.nSamples(selDim) = numel(selArray);
        dimInfoStruct.resolutions(selDim) = ...
            uniqueDistance.*dimInfoStruct.resolutions(selDim);
        dimInfoStruct.samplingPoints{selDim} = ...
            dimInfoStruct.samplingPoints{selDim}(selArray);
        dimInfoStruct.ranges(:,selDim) = ...
            [min(dimInfoStruct.samplingPoints{selDim}), ...
            max(dimInfoStruct.samplingPoints{selDim})];
        
        actSolution = selectionDimInfoStruct;
        expSolution = dimInfoStruct;
        
    case 'multipleDimsWithSelection'
        
        % specify multiple selection dims
        selDim = [2, 3, 5];
        selDimChar = dimInfo.dimLabels(selDim);
        selArray = {1:30, [10,13,16], 2};
        
        % test if constant sampling interval is specified here (since
        % resolutions should not be set to nan)
        indexDistance = cellfun(@diff, selArray, 'UniformOutput', false);
        uniqueDistance = cellfun(@unique, indexDistance, 'UniformOutput', false);
        nUniqueDistances = cellfun(@numel,uniqueDistance);
        isConstantSampling = ~any(nUniqueDistances > 1);
        this.assertTrue(isConstantSampling, ...
            'Selection Array with uneuqal distance specified');
        for n = 1:numel(selDim)
            selection.(selDimChar{n}) = selArray{n};
        end
        % add selection type as index
        selection.type = 'index';
        % select using array indices
        selectionDimInfo = dimInfo.select(selection);
        
        % make both dimInfos to struct to make sure we are not calling
        % select again
        warning('off', 'MATLAB:structOnObject');
        dimInfoStruct = struct(dimInfo);
        selectionDimInfoStruct = struct(selectionDimInfo);
        warning('on', 'MATLAB:structOnObject');
        
        % select in struct
        % nSamples
        dimInfoStruct.nSamples(selDim) = cellfun(@numel, selArray);
        % resolutions
        uniqueDistanceEmpty = cellfun(@isempty, uniqueDistance);
        for nUnique = 1:numel(uniqueDistanceEmpty)
            if ~uniqueDistanceEmpty(nUnique)
                uniqueDistanceForRes(nUnique) = uniqueDistance{nUnique};
            else
                uniqueDistanceForRes(nUnique) = 0;
            end
        end
        dimInfoStruct.resolutions(selDim) = ...
            uniqueDistanceForRes.* dimInfoStruct.resolutions(selDim);
        % sampling Points
        for nSamplingPoints = 1:numel(selDim)
            dimInfoStruct.samplingPoints{selDim(nSamplingPoints)} = ...
                dimInfoStruct.samplingPoints{selDim(nSamplingPoints)}(selArray{nSamplingPoints});
        end
        
        % ranges
        dimInfoStruct.ranges(1,selDim) = ...
            cellfun(@min, dimInfoStruct.samplingPoints(selDim));
        dimInfoStruct.ranges(2,selDim) = ...
            cellfun(@max, dimInfoStruct.samplingPoints(selDim));
        
        % define actual and expected solution
        expSolution = dimInfoStruct;
        actSolution = selectionDimInfoStruct;
        
    case 'type'
        selDim = 3;
        selDimChar = dimInfo.dimLabels{selDim};
        selArray = [10,13,16];
        
        % select using array indices
        selectionDimInfoIndex = ...
            dimInfo.select('type', 'index', selDimChar, selArray);
        
        % make both selectionDimInfoIndex to struct to make sure select is
        % not called again
        warning('off', 'MATLAB:structOnObject');
        selectionDimInfoIndexStruct = struct(selectionDimInfoIndex);
        warning('on', 'MATLAB:structOnObject');
        
        % get selected samples and select based on samples
        selSamples = selectionDimInfoIndexStruct.samplingPoints{selDim};
        selectionDimInfoSample = ...
            dimInfo.select('type', 'sample', selDimChar, selSamples);
        
        % make struct to allows comparison
        warning('off', 'MATLAB:structOnObject');
        actSolution = struct(selectionDimInfoSample);
        warning('on', 'MATLAB:structOnObject');
        % actual solution
        expSolution = selectionDimInfoIndexStruct;
        
    case 'invert'
        selDim = 3;
        selDimChar = dimInfo.dimLabels{selDim};
        selArray = [10,13,16];
        
        InvSelDim = 1:dimInfo.nDims;
        InvSelDimChar = dimInfo.dimLabels(InvSelDim);
        InvSelArray = dimInfo.samplingPoints;
        InvSelArray{selDim} = setdiff(InvSelArray{selDim}, ...
            InvSelArray{selDim}(selArray));
        for nInvSelDims = 1:dimInfo.nDims
            selection.(InvSelDimChar{nInvSelDims}) = ...
                InvSelArray{nInvSelDims};
        end
        selection.type = 'sample';       
        % select DimInfo
        selectionDimInfo = dimInfo.select(selDimChar, selArray, 'invert', 'true');
        selectionDimInfoInvert = dimInfo.select(selection);
        
        % make struct to allows comparison
        warning('off', 'MATLAB:structOnObject');
        actSolution = struct(selectionDimInfo);
        expSolution = struct(selectionDimInfoInvert);
        warning('on', 'MATLAB:structOnObject');
        
    case 'removeDims'
        % select used the first voxel over time
        tDim = 4;
        selDim = [1:3, 5];
        selDimChar = dimInfo.dimLabels(selDim);
        
        selectionDimInfo = dimInfo.select('removeDims', true, ...
            selDimChar{1}, 1, selDimChar{2}, 1, selDimChar{3}, 1, ...
            selDimChar{4}, 1);
        
        % make struct to allows comparison
        warning('off', 'MATLAB:structOnObject');
        actSolution = struct(selectionDimInfo);
        dimInfoStruct = struct(dimInfo);
        warning('on', 'MATLAB:structOnObject');
        expSolution.nDims = 1;
        expSolution.nSamples = dimInfoStruct.nSamples(tDim);
        expSolution.resolutions = dimInfoStruct.resolutions(tDim);
        expSolution.ranges = dimInfoStruct.ranges(:, tDim);
        expSolution.dimLabels = dimInfoStruct.dimLabels(tDim);
        expSolution.units = dimInfoStruct.units(tDim);
        expSolution.samplingPoints = dimInfoStruct.samplingPoints(tDim);
        expSolution.samplingWidths = dimInfoStruct.samplingWidths(tDim);
        
    case 'unusedVarargin'
        % same test as 'singleDim', but with unused varargin
        selDim = 3;
        selDimChar = dimInfo.dimLabels{selDim};
        selArray = [10,13,16];
        % test if constant sampling interval is specified here (since we
        % want to have resolutions)
        indexDistance = diff(selArray);
        uniqueDistance = unique(indexDistance);
        isConstantSampling = numel(uniqueDistance) == 1;
        this.assertTrue(isConstantSampling, ...
            'Selection Array with uneuqal distance specified');
        % select DimInfo
        [selectionDimInfo, ~, unusedVarargin] = ...
            dimInfo.select(selDimChar, selArray, 'giveBack', 'unusedVarargin');
        
        % make both dimInfos to struct to make sure we are not calling
        % select again
        warning('off', 'MATLAB:structOnObject');
        dimInfoStruct = struct(dimInfo);
        selectionDimInfoStruct = struct(selectionDimInfo);
        warning('on', 'MATLAB:structOnObject');
        
        % select in struct
        dimInfoStruct.nSamples(selDim) = numel(selArray);
        dimInfoStruct.resolutions(selDim) = ...
            uniqueDistance.*dimInfoStruct.resolutions(selDim);
        dimInfoStruct.samplingPoints{selDim} = ...
            dimInfoStruct.samplingPoints{selDim}(selArray);
        dimInfoStruct.ranges(:,selDim) = ...
            [min(dimInfoStruct.samplingPoints{selDim}), ...
            max(dimInfoStruct.samplingPoints{selDim})];
        
        actSolution.dimInfo = selectionDimInfoStruct;
        actSolution.Varargin = unusedVarargin;
        expSolution.dimInfo = dimInfoStruct;
        expSolution.Varargin = {'giveBack', 'unusedVarargin'};
        
end

% verify is actual solution matches expected solution
this.verifyEqual(actSolution, expSolution, 'absTol', 10e-7);

end

