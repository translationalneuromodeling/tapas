classdef MrUnitTest < matlab.unittest.TestCase
    % Implements unit testing for all MrClasses
    %
    % EXAMPLE
    %   MrUnitTest
    %
    %   See also
    
    % Author:   Saskia Bollmann
    % Created:  2017-07-07
    % Copyright (C) 2017 Institute for Biomedical Engineering
    %                    University of Zurich and ETH Zurich
    %
    % This file is part of the TAPAS UniQC Toolbox, which is released
    % under the terms of the GNU General Public License (GPL), version 3.
    % You can redistribute it and/or modify it under the terms of the GPL
    % (either version 3 or, at your option, any later version).
    % For further details, see the file COPYING or
    %  <http://www.gnu.org/licenses/>.
    
    properties (TestParameter)
        % MrDimInfo
        testVariantsDimInfo = {'1', '2', '3', '4', '5', '6', '7', '8', '9', ...
            '10', '11'};
        testFile = {'3DNifti', '4DNifti', 'Folder', 'ParRec'};
        testVariantsDimInfoSplit = {'singleDim', 'twoDims', 'nonExistingDim', ...
            'charSplitDim', 'differentIndex', 'cellofChars', 'emptySplitDim'};
        testVariantsDimInfoSelect = {'singleDim', 'multipleDimsWithSelection',...
            'type', 'invert', 'removeDims', 'unusedVarargin'};
        testCaseLoadMat = {'checkTempDir', 'oneVar', 'objectAsStruct', ...
            'className', 'noMatch', 'tooManyMatch', 'withVarName'};
        % MrAffineTransformation
        testVariantsAffineTrafo = {'propVal', 'matrix'};
        testFileAffineTrafo = {'3DNifti', '4DNifti', 'ParRec'};
        % MrImageGeometry
        testVariantsImageGeom = {'makeReference', ...
            'dimInfoAndaffineTransformation', 'matrix', 'dimInfo', ...
            'affineTransformation', 'timing_info'};
        % MrDataNd
        testVariantsDataNd = {'matrix', 'matrixWithDimInfo', 'matrixWithPropVal'};
        testVariantsArithmeticOperation = {'minus', 'plus', 'power', 'rdivide', 'times'};
        testVariantsDimensionOperation = {'circshift', 'flip', 'fliplr', 'flipud', ...
            'resize', 'rot90', 'select', 'split'};
        testVariantsValueOperation = {'cumsum', 'diff', 'fft', 'hist', 'ifft', ...
            'isreal', 'max', 'maxip', 'mean', 'power', 'prctile', 'real', ...
            'rms', 'rmse', 'unwrap'};
        testVariantsShiftTimeseries = {0}; % verbosity level (plots)
        testVariantsSelect = {'multipleDims', 'invert', 'removeDims', 'unusedVarargin'};
        % MrImage
        MrImageLoadConditions = {'4DNifti', 'FilePlusDimLabelsUnits', ...
            'FilePlusResolutions', 'FilePlussamplingWidths', ...
            'FilePlusSamplingPoints', 'FilePlusShearRotation', ...
            'FilePlusSelect', 'FilePlusDimInfoPropVals', ...
            'FilePlusAffineTransformation', 'FilePlusFirstSamplingPoint', ...
            'FilePlusLastSamplingPoint', 'FilePlusArrayIndex', ...
            'FilePlusOriginIndex'};
    end
    %% MrDimInfo
    methods (Test, TestTags = {'Constructor', 'MrDimInfo'})
        this = MrDimInfo_constructor(this, testVariantsDimInfo)
        this = MrDimInfo_load_from_file(this, testFile)
        this = MrDimInfo_load_from_mat(this, testCaseLoadMat)
    end
    
    methods (Test, TestTags = {'Methods', 'MrDimInfo'})
        this = MrDimInfo_get_add_remove(this)
        this = MrDimInfo_index2sample(this)
        this = MrDimInfo_permute(this)
        this = MrDimInfo_split(this, testVariantsDimInfoSplit)
        this = MrDimInfo_select(this, testVariantsDimInfoSelect)
        this = MrDimInfo_update_and_validate_properties_from(this)
    end
    
    %% MrAffineTransformation
    methods (Test, TestTags = {'Constructor', 'MrAffineTransformation'})
        this = MrAffineTransformation_constructor(this, testVariantsAffineTrafo)
        this = MrAffineTransformation_load_from_file(this, testFileAffineTrafo)
    end
    
    methods (Test, TestTags = {'Methods', 'MrAffineTransformation'})
        this = MrAffineTransformation_transformation(this)
        this = MrAffineTransformation_affineMatrix(this)
    end
    %% MrImageGeometry
    methods (Test, TestTags = {'Constructor', 'MrImageGeometry'})
        this = MrImageGeometry_constructor(this, testVariantsImageGeom)
        this = MrImageGeometry_load_from_file(this, testFile)
    end
    
    methods (Test, TestTags = {'Methods', 'MrImageGeometry'})
        this = MrImageGeometry_create_empty_image(this)
    end
    %% MrDataNd
    methods (Test, TestTags = {'Constructor', 'MrDataNd'})
        this = MrDataNd_constructor(this, testVariantsDataNd)
        % loading of nifti data will be tested in MrImage (since there,
        % also the affineTransformation is created)
    end
    
    methods (Test, TestTags = {'Methods', 'MrDataNd'})
        this = MrDataNd_arithmetic_operation(this, testVariantsArithmeticOperation);
        this = MrDataNd_permute(this);
        this = MrDataNd_select(this, testVariantsSelect);
        % this = MrDataNd_dimension_operation(this, testDimensionOperation);
        this = MrDataNd_value_operation(this, testVariantsValueOperation);
        this = MrDataNd_shift_timeseries(this, testVariantsShiftTimeseries);
        % these take too long, need to be shortened
        % this = MrDataNd_split_epoch(this);
    end
    
    %% MrImage
    methods (Test, TestTags = {'Constructor', 'MrImage'})
        this = MrImage_load_from_file(this, MrImageLoadConditions)
    end
    
end
