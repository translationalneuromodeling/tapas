function [fileNameArray, fileNameDeletedArray] = tapas_uniqc_delete_with_hdr(fileNameArray)
% Deletes given files; for .nii (nifti-) files, also deletes .mat-header,
% and _dimInfo.mat (if existing);
% for given .img (analyze) files, also deletes .hdr-header and _dimInfo.mat
%
%   fileNameArray = tapas_uniqc_delete_with_hdr(fileNameArray)
%
% IN
%   fileNameArray   cell of filenames to be deleted
% OUT
%   fileNameArray   cell of filenames that were tried to be deleted
%                   (includes .mat files corresponding to .nii
%                    .hdr files corresponding to .img)
%   fileNameDeletedArray
%                   cell of filenames that were actually deleted (existed)
%
% EXAMPLE
%   tapas_uniqc_delete_with_hdr('temp.nii');
%
%   See also move_with_mat

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-08
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


if ~isempty(fileNameArray) % only do sth, if files are given, no '' or {}
    if ~iscell(fileNameArray)
        fileNameArray = cellstr(fileNameArray);
    end
    
    fileNameArray = fileNameArray(:);
    
    % append all .mat files to list of deletable files that corresponding to .nii
    % append all .hdr files to list of deletable files that corresponding to .img
    % append all _dimInfo.mat files to list of deletable files that corresponding to .nii
    % append all _dimInfo.mat files to list of deletable files that corresponding to .img

    iImgFiles = find(~cellfun(@isempty, regexp(fileNameArray, '\.img$')));
    iNiftiFiles = find(~cellfun(@isempty, regexp(fileNameArray, '\.nii$')));
    fileNameArray = [
        fileNameArray
        regexprep(fileNameArray(iNiftiFiles), '\.nii$', '\.mat')
        regexprep(fileNameArray(iImgFiles), '\.img$', '\.hdr')
        regexprep(fileNameArray(iNiftiFiles), '\.nii$', '\_dimInfo.mat')
        regexprep(fileNameArray(iImgFiles), '\.img$', '\_dimInfo.mat')
        ];
    
    iExistingFiles = find(cell2mat(cellfun(@(x) exist(x, 'file'), fileNameArray, ...
        'UniformOutput', false)));
    fileNameDeletedArray = fileNameArray(iExistingFiles);
    delete(fileNameDeletedArray{:});
end
