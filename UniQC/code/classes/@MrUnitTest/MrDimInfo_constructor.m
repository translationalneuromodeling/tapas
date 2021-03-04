function this = MrDimInfo_constructor(this, dimInfoVariants)
% Unit test for MrDimInfo Constructor evoking all 6 variants
%
%   Y = MrUnitTest()
%   run(Y, 'MrDimInfo_constructor');
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDimInfo_constructor
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2017-08-08
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


% Unit test for MrDimInfo Constructor evoking all 6 variants via
% dimInfoVariants

% construct MrDimInfo object from sampling points - this is the actual
% solution for variant (1) and the expected solution for all other cases
% using make_dimInfo_reference
dimInfo = this.make_dimInfo_reference(0);

switch dimInfoVariants
    %% (1) explicit setting of sampling points for dimension(s)
    case '1'
        
        % define actual solution
        actSolution = dimInfo;
        % load expected solution
        classesPath = tapas_uniqc_get_path('classes');
        solutionFileName = fullfile(classesPath, '@MrUnitTest' , 'dimInfo.mat');
        expSolution = load(solutionFileName);
        expSolution = expSolution.dimInfo;
        
        %% (2) nSamples + ranges
    case '2'
        % define expected solution
        expSolution = dimInfo;
        % define actual solution
        % Initialize MrDimInfo via nSamples + ranges (variant (2))
        actSolution = MrDimInfo(...
            'dimLabels', expSolution.dimLabels, ...
            'units', expSolution.units, ...
            'nSamples', expSolution.nSamples, ...
            'ranges', expSolution.ranges);
        % overwrite samplingPoints of coil due to non-uniform sampling
        actSolution.coil.samplingPoints = expSolution.coil.samplingPoints;
        % overwrite samplingWidths since they are retained in case of later
        % changes, but not set because of the construction of the reference
        % MrDimInfo object (= expected Solution)
        actSolution.coil.samplingWidths = nan;
        
        
        %% (3) nSamples + resolutions + samplingPoint + arrayIndex
    case '3'
        % define expected solution
        expSolution = dimInfo;
        % define actual solution
        arrayIndex = [81 54 14 321 1];
        
        actSolution = MrDimInfo(...
            'dimLabels', expSolution.dimLabels, ...
            'units', expSolution.units, ...
            'nSamples', expSolution.nSamples, ...
            'resolutions', expSolution.resolutions, ...
            'samplingPoint', expSolution.index2sample(arrayIndex), ....
            'arrayIndex', arrayIndex);
        
        % overwrite samplingPoints of coil due to non-uniform sampling
        actSolution.coil.samplingPoints = expSolution.coil.samplingPoints;
        
        % overwrite samplingWidths since they are retained in case of later
        % changes, but not set because of the construction of the reference
        % MrDimInfo object (= expected Solution)
        actSolution.coil.samplingWidths = nan;
        
        %% (4) nSamples + resolutions + firstSamplingPoint
    case '4'
        % define expected solution
        expSolution = dimInfo;
        % define actual solution
        arrayIndex = [1 1 1 1 1];
        
        actSolution = MrDimInfo(...
            'dimLabels', expSolution.dimLabels, ...
            'units', expSolution.units, ...
            'nSamples', expSolution.nSamples, ...
            'resolutions', expSolution.resolutions, ...
            'firstSamplingPoint', expSolution.index2sample(arrayIndex));
        
        % overwrite samplingPoints of coil due to non-uniform sampling
        actSolution.coil.samplingPoints = expSolution.coil.samplingPoints;
        
        % overwrite samplingWidths since they are retained in case of later
        % changes, but not set because of the construction of the reference
        % MrDimInfo object (= expected Solution)
        actSolution.coil.samplingWidths = nan;
        
        
        %% (5) nSamples + resolutions + lastSamplingPoint
    case '5'
        % define expected solution
        expSolution = dimInfo;
        % define actual solution
        arrayIndex = expSolution.nSamples;
        
        actSolution = MrDimInfo(...
            'dimLabels', expSolution.dimLabels, ...
            'units', expSolution.units, ...
            'nSamples', expSolution.nSamples, ...
            'resolutions', expSolution.resolutions, ...
            'lastSamplingPoint', expSolution.index2sample(arrayIndex));
        
        % overwrite samplingPoints of coil due to non-uniform sampling
        actSolution.coil.samplingPoints = expSolution.coil.samplingPoints;
        
        % overwrite samplingWidths since they are retained in case of later
        % changes, but not set because of the construction of the reference
        % MrDimInfo object (= expected Solution)
        actSolution.coil.samplingWidths = nan;
        
    case '6'
        %% (6) nSamples only
        % get nSamples from reference object
        nSamples = dimInfo.nSamples;
        nDims = numel(nSamples);
        
        % make actual solution
        actSolution = MrDimInfo('nSamples', nSamples);
        
        % make expected solution from ranges and sampling points
        
        for n = 1:nDims
            if n < 4
                % x y z
                samplingPoints{n} = -(nSamples(n)-1)/2:(nSamples(n)-1)/2;
            else
                samplingPoints{n} = 1:(nSamples(n));
            end
        end
        expSolution = MrDimInfo('samplingPoints', samplingPoints);
        
    case '7'
        %% (7) resolutions only
        % get resolutions from reference object
        resolutions = dimInfo.resolutions;
        % use only valid (i.e. finite)
        validResolutions = isfinite(resolutions);
        nDims = sum(validResolutions);
        
        % make actual solution
        actSolution = MrDimInfo('resolutions', resolutions(validResolutions));
        
        % make expected solution
        for n = 1:nDims
            if n < 4
                % x y z
                samplingPoints{n} = -(resolutions(n))/2:resolutions(n):(resolutions(n))/2;
            else
                samplingPoints{n} = resolutions(n):resolutions(n):2*resolutions(n);
            end
        end
        
        expSolution = MrDimInfo('samplingPoints', samplingPoints);
        
    case '8'
        %% (8) ranges only
        
        % get ranges from reference object
        ranges = dimInfo.ranges;
        
        % make actual solution
        actSolution = MrDimInfo('ranges', ranges);
        
        % make expected solution from sampling points
        samplingPoints = mat2cell(ranges', [1 1 1 1 1], 2)';
        expSolution = MrDimInfo('samplingPoints', samplingPoints);
        
    case '9'
        %% (9) dimLabels only
        
        % get dimLabels from reference object
        dimLabels = dimInfo.dimLabels;
        nDims = numel(dimLabels);
        
        % make actual solution
        actSolution = MrDimInfo('dimLabels', dimLabels);
        
        % make expected solution with standard sampling points, i.e [-0.5
        % 0.5] for dims x,y,z and [1 2] for all other dims
        
        expSolution.nDims = nDims;
        expSolution.nSamples = 2*ones(1, nDims);
        expSolution.resolutions = ones(1, nDims);
        expSolution.ranges = [-0.5 -0.5 -0.5 1 1; 0.5 0.5 0.5 2 2];
        for n = 1:nDims
            expSolution.dimLabels{n} = dimInfo.get_default_dim_labels(n);
            expSolution.units{n} = dimInfo.get_default_dim_units(n);
        end
        expSolution.samplingPoints = {[-0.5 0.5] [-0.5 0.5] [-0.5 0.5] [1 2] [1 2]};
        expSolution.samplingWidths = expSolution.resolutions;
        
    case '10'
        %% (10) units only
        
        % get units from reference object
        units = dimInfo.units;
        nDims = numel(units);
        
        % make actual solution
        actSolution = MrDimInfo('units', units);
        
        % make expected solution with standard sampling points, i.e [-0.5
        % 0.5] for dims x,y,z and [1 2] for all other dims
        expSolution.nDims = nDims;
        expSolution.nSamples = 2*ones(1, nDims);
        expSolution.resolutions = ones(1, nDims);
        expSolution.ranges = [-0.5 -0.5 -0.5 1 1; 0.5 0.5 0.5 2 2];
        for n = 1:nDims
            expSolution.dimLabels{n} = dimInfo.get_default_dim_labels(n);
            expSolution.units{n} = dimInfo.get_default_dim_units(n);
        end
        expSolution.samplingPoints = {[-0.5 0.5] [-0.5 0.5] [-0.5 0.5] [1 2] [1 2]};
        expSolution.samplingWidths = expSolution.resolutions;
        
    case '11'
        %% (11) samplingWidths only
        % note that samplingWidths does not impact resolutions and, thus,
        % sampling points and can therefore also be nan
        
        % get samplingWidths from reference object
        samplingWidths = dimInfo.samplingWidths;
        nDims = numel(samplingWidths);
        
        % make actual solution
        actSolution = MrDimInfo('samplingWidths', samplingWidths);
        
        % make expected solution with standard sampling points, i.e [-0.5
        % 0.5] for dims x,y,z and [1 2] for all other dims
        expSolution.nDims = nDims;
        expSolution.nSamples = 2*ones(1, nDims);
        expSolution.resolutions = ones(1, nDims);
        expSolution.ranges = [-0.5 -0.5 -0.5 1 1; 0.5 0.5 0.5 2 2];
        for n = 1:nDims
            expSolution.dimLabels{n} = dimInfo.get_default_dim_labels(n);
            expSolution.units{n} = dimInfo.get_default_dim_units(n);
        end
        expSolution.samplingPoints = {[-0.5 0.5] [-0.5 0.5] [-0.5 0.5] [1 2] [1 2]};
        expSolution.samplingWidths = samplingWidths;
        
end

% verify whether expected and actual solution are identical
% Note: convert to struct, since the PublicPropertyComparator (to allow
% nans to be treated as equal) does not compare properties of objects that
% overload subsref

warning('off', 'MATLAB:structOnObject');
this.verifyEqual(struct(actSolution), struct(expSolution), 'absTol', 10e-7);
warning('on', 'MATLAB:structOnObject');
end