function this = read_nifti_analyze(this, fileName, selectedVolumes)
% loads matrix into .data from nifti or analyze file using spm_read_vols
%
%   this = read_nifti_analyze(this, fileName, selectedVolumes)
%
% IN
%   fileName
%   selectedVolumes     [1,nVols] index of selected volumes to load,
%                       Inf for all; default: Inf
% OUT
%
% EXAMPLE
%   read_nifti_analyze
%
%   See also

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-04-16
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
    fileName = fullfile(this.parameters.path, ...
        this.parameters.unprocessedFile);
end

fileNameVolArray = tapas_uniqc_get_vol_filenames(fileName);

V = spm_vol(strvcat(fileNameVolArray));

hasSelectedVolumes = nargin > 2 && ~any(isinf(selectedVolumes));

if hasSelectedVolumes
    V = V(selectedVolumes);
end

% more than 4 dimensions not handles by spm_vol, read directly from
% file_array
if numel(V(1).private.dat.dim) > 4
    this.data = reshape(V.private.dat(:), size(V.private.dat));
else
    % use inbuilt SPM functionality
    try
        % this.data = transform_matrix_analyze2matlab(spm_read_vols(V));
        this.data = spm_read_vols(V);
        % maybe only header misalignment of volumes is the problem for nifti
        %...rename temporarily for loading
    catch err
        fnHdr = regexprep(fileName, '\.nii','\.mat');
        fnTmp = regexprep(fileName, '\.nii','\.tmp');
        if exist(fnHdr, 'file')
            movefile(fnHdr, fnTmp);
            warning('Headers of volumes not aligned, ignoring them...');
            V = spm_vol(fileName);
            this.data = spm_read_vols(V);
            movefile(fnTmp, fnHdr);
        else % nothing we can do, throw error
            throw(err);
        end
    end
end