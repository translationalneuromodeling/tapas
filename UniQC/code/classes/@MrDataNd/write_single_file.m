function filename = write_single_file(this, filename, dataType)
% saves Nd-data to a single file in different file formats, depending on extension
%
%   Y = MrDataNd();
%   filename = Y.write_single_file(filename)
%
% This is a method of class MrDataNd.
%
% NOTE: This is a low-level function to write to disk. For a high-level,
% felxible framework, use and see also MrDataNd.save.
%
% IN
%   filename    possible extensions:
%                   '.nii' - nifti
%                   '.img' - analyse, one file/scan volume
%                   '.mat' - save data and parameters separately
%                            export to matlab-users w/o class def files
%               default: parameters.save.path/parameters.save.fileUnprocessed
%               can be set via parameters.save.path.whichFilename = 0 to
%               parameters.save.path/parameters.save.fileName
%   dataType    number format for saving voxel values; see also spm_type
%               specified as one of the following string identifiers
%                'uint8','int16','int32','float32','float64','int8','uint16','uint32';
%               default (3D): single
%               default (4D or size > 30 MB): int32
%
% OUT
%
% EXAMPLE
%   write_single_file
%
%   See also MrDataNd MrDataNd.save

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-02
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if nargin < 2
    filename = this.get_filename();
end

if nargin < 3
    dataType = tapas_uniqc_get_data_type_from_n_voxels(this.dimInfo.nSamples);
end

% no data, no saving...
if isempty(this.data)
    fprintf('No data in MrDataNd %s; file %s not saved\n', this.name, ...
        filename);
    
else
    
    [fp, fn, ext] = fileparts(filename);
    
    if ~isempty(fp)
        [s, mess, messid] = mkdir(fp); % to suppress dir exists warning
    end
    
    switch ext
        case '.mat'
            % TODO: replace via obj2struct in MrCopyData and save as struct
            obj = this;
            save(filename, 'obj');
        case {'.nii', '.img', '.hdr'}
            this = write_nifti_analyze(this, filename, dataType);
        
            [fp, fn, ext] = fileparts(filename);
            fileNameDimInfo = fullfile(fp, [fn '_dimInfo.mat']);
            this.dimInfo.save(fileNameDimInfo);
        otherwise
            error('tapas:uniqc:MrDataNd:UnsuppportedFileType', 'Unsupported file type');
    end
end
