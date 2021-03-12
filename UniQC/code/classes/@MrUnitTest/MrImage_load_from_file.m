function this = MrImage_load_from_file(this, testCondition)
% Test loading from files for MrImage, in particular if additional
% parameters are given.
%
%   Y = MrUnitTest()
%   run(Y, MrImage_load_from_file)
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrImage_load_from_file
%
%   See also MrUnitTest

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-11-07
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

dataPath = tapas_uniqc_get_path('data');
niftiFile4D = fullfile(dataPath, 'nifti', 'rest', 'fmri_short.nii');

switch testCondition
    case '4DNifti'
        % 4D Nifti
        % actual solution
        image = MrImage(niftiFile4D);
        % check poperties first
        % dimInfo
        % - assumes MrDimInfo(fileName) works and just verifies
        % that dimInfo is properly added to the object
        refDimInfo = MrDimInfo(niftiFile4D);
        if ~image.dimInfo.isequal(refDimInfo)
            this.assertFail('Failed to load or update dimInfo for MrImage.');
        end
        
        % imageGeometry
        % - assumes MrImageGeometry(fileName) works and just verifies
        % that imageGeometry is properly added to the object
        refImageGeom = MrImageGeometry(niftiFile4D);
        if ~image.geometry.isequal(refImageGeom)
            % the imageGeometry is created from dimInfo and affineTrafo, so
            % check MrAffineTrafo
            refAffineTrafo = MrAffineTransformation(niftiFile4D, image.dimInfo);
            if ~image.affineTransformation.isequal(refAffineTrafo)
                this.assertFail('Failed to load or update affineTrafo for MrImage.');
            else
                this.assertFail('Failed to compute imageGeometry for MrImage.');
            end
        end
        
        % data
        % - we don't want to save the whole matrix, so we just compare a
        % pre-computed hash
        md = java.security.MessageDigest.getInstance('MD5');
        actSolution = sprintf('%2.2x', typecast(md.digest(image.data(:)), 'uint8')');
        % pre-computed hash:
        expSolution = 'cc9db2c532989fc1b6585c38e2c66e68';
        
    case 'FilePlusDimLabelsUnits'
        % check whether labels and units are correctly passed along
        % actual solution
        image = MrImage(niftiFile4D, 'dimLabels', {'dL1', 'dL2', 'dL3', 'dL4'}, ...
            'units', {'u1', 'u2', 'u3', 'u4'});
        actSolution = image.dimInfo;
        expSolution = MrDimInfo(niftiFile4D);
        expSolution.set_dims(1:4,'dimLabels', {'dL1', 'dL2', 'dL3', 'dL4'}, ...
            'units', {'u1', 'u2', 'u3', 'u4'});
        
    case 'FilePlusResolutions'
        % check if resolutions are adapted accordingly
        image = MrImage(niftiFile4D, 'resolutions', [1.3 5 0.4 2]);
        actSolution = image.dimInfo;
        expSolution = MrDimInfo(niftiFile4D);
        expSolution.resolutions = [1.3 5 0.4 2];
        
    case 'FilePlussamplingWidths'
        % check if samplingWidths are adapted accordingly
        image = MrImage(niftiFile4D, 'samplingWidths', [1.3 5 0.4 2]);
        actSolution = image.dimInfo;
        expSolution = MrDimInfo(niftiFile4D);
        expSolution.samplingWidths = [1.3 5 0.4 2];
        
    case 'FilePlusSamplingPoints'
        % check if samplingPoints are adapted accordingly
        dimInfo = MrDimInfo(niftiFile4D);
        samplingPoints = {1:dimInfo.nSamples(1), 1:dimInfo.nSamples(2), ...
            1:dimInfo.nSamples(3), 1:dimInfo.nSamples(4)};
        image = MrImage(niftiFile4D, 'samplingPoints', samplingPoints);
        actSolution = image.dimInfo;
        expSolution = MrDimInfo(niftiFile4D);
        expSolution.samplingPoints = samplingPoints;
        
    case 'FilePlusShearRotation'
        % check if MrAffineGeometry is adapted accordingly
        image = MrImage(niftiFile4D, 'shear', [0 0.5 0], ...
            'rotation_deg', [0 30 67]);
        actSolution = image.affineTransformation;
        expSolution = MrAffineTransformation(niftiFile4D, image.dimInfo);
        expSolution.shear = [0 0.5 0];
        expSolution.rotation_deg = [0 30 67];
        
    case 'FilePlusSelect'
        select.z = 20;
        actSolution = MrImage(niftiFile4D, 'select', select);
        expSolution = MrImage(niftiFile4D).select(select);
        % clear parameters' save path (objects were created at different
        % times)
        actSolution.parameters.save.path = '';
        expSolution.parameters.save.path = '';
        
    case 'FilePlusDimInfoPropVals'
        args = {'dimLabels', {'dL1', 'dL2', 'dL3', 'dL4'}, ...
            'units',{'u1', 'u2', 'u3', 'u4'}, ...
            'samplingWidths', [1.3 5 0.4 2]};
        actSolution = MrImage(niftiFile4D, args{:});
        actSolution.parameters.save.path = '';
        expSolution = MrImage(niftiFile4D);
        expSolution.dimInfo.set_dims(1:4, args{:});
        expSolution.parameters.save.path = '';
        
    case 'FilePlusAffineTransformation'
        expSolution = this.make_affineTransformation_reference(0);
        actSolution = MrImage(niftiFile4D, ...
            'affineMatrix', expSolution.affineMatrix);
        actSolution = actSolution.affineTransformation;
        
    case 'FilePlusFirstSamplingPoint'
        firstSamplingPoint = [0 2 -5 0.8];
        m = MrImage(niftiFile4D, 'firstSamplingPoint', firstSamplingPoint);
        actSolution = [m.dimInfo.samplingPoints{1}(1), m.dimInfo.samplingPoints{2}(1), ...
            m.dimInfo.samplingPoints{3}(1), m.dimInfo.samplingPoints{4}(1)];
        expSolution = firstSamplingPoint;
        
    case 'FilePlusLastSamplingPoint'
        lastSamplingPoint = [0 2 -5 0.8];
        m = MrImage(niftiFile4D, 'lastSamplingPoint', lastSamplingPoint);
        actSolution = [m.dimInfo.samplingPoints{1}(end), m.dimInfo.samplingPoints{2}(end), ...
            m.dimInfo.samplingPoints{3}(end), m.dimInfo.samplingPoints{4}(end)];
        expSolution = lastSamplingPoint;
        
    case 'FilePlusArrayIndex'
        samplingPoints = [0 2 -5 0.8];
        arrayIndex = [56 13 7 4];
        m = MrImage(niftiFile4D, 'arrayIndex', arrayIndex, ...
            'samplingPoint', samplingPoints);
        actSolution = [m.dimInfo.samplingPoints{1}(arrayIndex(1)), m.dimInfo.samplingPoints{2}(arrayIndex(2)), ...
            m.dimInfo.samplingPoints{3}(arrayIndex(3)), m.dimInfo.samplingPoints{4}(arrayIndex(4))];
        expSolution = samplingPoints;
        
    case 'FilePlusOriginIndex'
        originIndex = [56 13 7 4];
        m = MrImage(niftiFile4D, 'originIndex', originIndex);
        actSolution = [m.dimInfo.samplingPoints{1}(originIndex(1)), m.dimInfo.samplingPoints{2}(originIndex(2)), ...
            m.dimInfo.samplingPoints{3}(originIndex(3)), m.dimInfo.samplingPoints{4}(originIndex(4))];
        expSolution = zeros(1,4);
end

% verify equality
if isa(expSolution, 'MrDimInfo')
    warning('off', 'MATLAB:structOnObject');
    this.verifyEqual(struct(actSolution), struct(expSolution), 'absTol', 10e-7);
    warning('on', 'MATLAB:structOnObject');
else
    this.verifyEqual(actSolution, expSolution);
end
end
