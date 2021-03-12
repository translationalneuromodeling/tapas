function fileNameTemp = write_temporary_nifti_for_spm(this)
% saves the object to a nifti file in a unique temporary location, e.g.,
% for display in SPM
%
%   Y = MrDataNd()
%   fileNameTemp = Y.write_temporary_nifti_for_spm()
%
% This is a method of class MrDataNd.
%
% IN
%
% OUT
%   fileNameTemp    unique, temporary file name in same path as this image
%                   e.g. /home/myself/MrImage.nii -> 
%                        /home/myself/MrImage_tpc58aa993_a5b9_428f_af06_4723f9c737e0.nii
% EXAMPLE
%   write_temporary_nifti_for_spm
%
%   See also MrDataNd
 
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-11-29
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

[~,fn] = fileparts(this.parameters.save.fileName);

fileNameTemp = [tapas_uniqc_prefix_files(tempname(this.parameters.save.path), ...
    [fn '_']) '.nii'];

% return value could be an array of filenames, if nifti is split
[~, fileNameTemp] = this.save('fileName', fileNameTemp);
