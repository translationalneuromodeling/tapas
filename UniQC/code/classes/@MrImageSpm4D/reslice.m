function reslicedImage = reslice(this, targetGeometry, varargin)
% Reslices image to image geometry of other image using spm_reslice
%   Y = MrImageSpm4D()
%   resliceImage = Y.reslice(targetGeometry, ...
%       'spmParameterName1', spmParameterValue1, ...
%       ...
%       'spmParameterNameN', spmParameterValueN)
%
%   OR
%   Y.reslice(otherImage);
%
% This is a method of class MrImageSpm4D.
%
% IN
%   targetGeometry  object of MrImageGeometry or MrImage
%                   Image will be resliced to this geometry
%   interpolation   degree of b-spline interpolation for estimation and reslicing
%                   default: 7
%   wrapping        fold-over direction (phase encode)
%                   default: [0 0 0] % none
%   masking         mask incomplete timeseries?
%                   default: true
%
%
% OUT
%
%   reslicedImage
%
% EXAMPLE
%   Y = MrImage();
%   Z = MrImage();
%   targetGeometry = Z.geometry;
%   Y.reslice(targetGeometry)
%
%   See also MrImage MrImageGeometry spm_reslice spm_run_coreg

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-18
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

defaults.interpolation = 7;     % degree of b-spline interpolation for estimation and reslicing
defaults.wrapping = [0 0 0];    % fold-over direction (phase encode)
defaults.masking = 1;           % mask incomplete timeseries?

args = tapas_uniqc_propval(varargin, defaults);
reslicedImage = this.copyobj();

% Save as nifti to use spm functionality
% but check if file already exists, give new filename to prevent
% accidental overwrite
if isnumeric(reslicedImage.parameters.save.keepCreatedFiles)
    keepCreatedFiles = reslicedImage.parameters.save.keepCreatedFiles;
else
    keepCreatedFiles = ~strcmpi(reslicedImage.parameters.save.keepCreatedFiles, 'none');
end

changeFilename = isfile(reslicedImage.get_filename) && ~keepCreatedFiles;
if changeFilename
    origFilename = reslicedImage.parameters.save.fileName;
    [~, tmpName] = fileparts(tempname);
    reslicedImage.parameters.save.fileName = [tmpName, '.nii'];
end

reslicedImage.save('fileName', reslicedImage.get_filename('prefix', 'raw'));

% check whether input is actually a geometry
isGeometry = isa(targetGeometry, 'MrImageGeometry');
if ~isGeometry
    if isa(targetGeometry, 'MrImage')
        targetGeometry = targetGeometry.geometry;
    else
        disp('Input has to be of class MrImage or MrImageGeometry.');
    end
end

[~, ~, isEqualGeom3D] = targetGeometry.diffobj(reslicedImage.geometry);

if ~isEqualGeom3D
    
    % Dummy 3D image with right geometry is needed for resizing
    emptyImage = targetGeometry.create_empty_image('z', 1);
    emptyImage.parameters.save.path = reslicedImage.parameters.save.path;
    emptyImage.save();
    fnTargetGeometryImage = emptyImage.get_filename;
    
    matlabbatch = reslicedImage.get_matlabbatch('reslice', fnTargetGeometryImage, args);
    save(fullfile(reslicedImage.parameters.save.path, 'matlabbatch.mat'), ...
        'matlabbatch');
    spm_jobman('run', matlabbatch);
    
    % clean up: move/delete processed spm files, load new data into matrix
    reslicedImage.finish_processing_step('reslice', fnTargetGeometryImage);
end
% set back to original filename
if changeFilename
    reslicedImage.parameters.save.fileName = origFilename;
end

end