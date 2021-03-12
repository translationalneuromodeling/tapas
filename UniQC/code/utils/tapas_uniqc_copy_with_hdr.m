function [fileNameSourceArray, fileNameTargetArray, ...
    fileNameSourceMovedArray, fileNameTargetMovedArray] = tapas_uniqc_copy_with_hdr(...
    fileNameSourceArray, fileNameTargetArray, doMove)
% Moves given files; for .nii (nifti-) files, also moves .mat-header,
% and _dimInfo.mat (if existing);
% for given .img (analyze) files, also moves .hdr-header and _dimInfo.mat
%
%   [fileNameSourceArray, fileNameTargetArray] = tapas_uniqc_copy_with_hdr(...
%    fileNameSourceArray, fileNameTargetArray)
%
% IN
%   fileNameSourceArray   cell of filenames to be moved (source)
%   fileNameTargetArray   cell of filenames to be moved to (targets)
%   doMove                default: false
%                         if true, a move of the files (instead of copying)
%                         is performed, i.e. existing sources are deleted
%
% OUT
%   fileNameSourceArray   cell of filenames that were tried to be moved
%                   (includes   .mat files corresponding to .nii and
%                               .hdr files corresponding to .img
%                               _dimInfo.mat files corresponding to .nii or .img)
%   fileNameSourceMovedArray   cell of source filenames that were actually moved
%                   (includes   .mat files corresponding to .nii and
%                               .hdr files corresponding to .img)
%                               _dimInfo.mat files corresponding to .nii or .img)
%   fileNameTargetMovedArray   cell of target filenames that were actually moved
%                   (includes   .mat files corresponding to .nii and
%                               .hdr files corresponding to .img)
%                               _dimInfo.mat files corresponding to .nii or .img)
%
% EXAMPLE
%   tapas_uniqc_copy_with_hdr('from.nii', 'to.nii')
%   tapas_uniqc_copy_with_hdr('from.img', 'to.hdr')
%
%   See also tapas_uniqc_delete_with_hdr tapas_uniqc_move_with_hdr

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

if nargin < 3
    doMove = false;
end

if doMove
    fnCopyMove = @movefile;
else
    fnCopyMove = @copyfile;
end

if ~iscell(fileNameSourceArray)
    fileNameSourceArray = cellstr(fileNameSourceArray);
end

if ~iscell(fileNameTargetArray)
    fileNameTargetArray = cellstr(fileNameTargetArray);
end

fileNameSourceArray = fileNameSourceArray(:);
fileNameTargetArray = fileNameTargetArray(:);

% append all .mat/.hdr files to list of copiable files that corresponding to
% .nii/.img
iImgFiles = find(~cellfun(@isempty, regexp(fileNameSourceArray, '\.img$')));
iNiftiFiles = find(~cellfun(@isempty, regexp(fileNameSourceArray, '\.nii$')));

fileNameSourceArray = [
    fileNameSourceArray
    regexprep(fileNameSourceArray(iNiftiFiles), '\.nii$', '\.mat')
    regexprep(fileNameSourceArray(iImgFiles), '\.img$', '\.hdr')
    regexprep(fileNameSourceArray(iNiftiFiles), '\.nii$', '\_dimInfo.mat')
    regexprep(fileNameSourceArray(iImgFiles), '\.img$', '\_dimInfo.mat')
    ];

fileNameTargetArray = [
    fileNameTargetArray
    regexprep(fileNameTargetArray(iNiftiFiles), '\.nii$', '\.mat')
    regexprep(fileNameTargetArray(iImgFiles), '\.img$', '\.hdr')
    regexprep(fileNameTargetArray(iNiftiFiles), '\.nii$', '\_dimInfo.mat')
    regexprep(fileNameTargetArray(iImgFiles), '\.img$', '\_dimInfo.mat')
    ];


iExistingFiles = find(cell2mat(cellfun(@(x) exist(x, 'file'), fileNameSourceArray, ...
    'UniformOutput', false)));
fileNameSourceMovedArray = fileNameSourceArray(iExistingFiles);
fileNameTargetMovedArray = fileNameTargetArray(iExistingFiles);

% copy all files one by one :-(
nFiles = numel(fileNameSourceMovedArray);
for iFile = 1:nFiles
    fnCopyMove(fileNameSourceMovedArray{iFile}, fileNameTargetMovedArray{iFile});
end