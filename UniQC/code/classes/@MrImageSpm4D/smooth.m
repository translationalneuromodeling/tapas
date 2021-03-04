function outputImage = smooth(this, varargin)
% Smoothes image (or image time series) spatially with sinc-truncated Gaussian kernel, 
% i.e., using spm_smooth and mimicking its functionality
%
%   Y = MrImageSpm4D()
%   sY = Y.smooth('fwhm', fwhm)
%
% outputImage is a method of class MrImageSpm4D.
% 
% NOTE: For complex-valued data, use MrImage.smooth
%
% IN
%   fwhm    [1,1] or [1,3] Full width at half maximum of Gaussian kernel (in mm)
%           if single value is given, isotropic kernel is assumed
%           default: 8
%
% OUT
%
% EXAMPLE
%   smooth
%
%   See also MrImageSpm4D MrImage.smooth spm_smooth

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-02
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% outputImage file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

defaults.fwhm = 8;

args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

if length(fwhm) == 1
    fwhm = fwhm*[ 1 1 1];
end

outputImage = this.copyobj;

% save image file for processing as nii in SPM
outputImage.save('fileName', outputImage.get_filename('prefix', 'raw'));

matlabbatch = outputImage.get_matlabbatch('smooth', fwhm);
save(fullfile(outputImage.parameters.save.path, 'matlabbatch.mat'), ...
            'matlabbatch');
spm_jobman('run', matlabbatch);

% clean up: move/delete processed spm files, load new data into matrix
outputImage.finish_processing_step('smooth');