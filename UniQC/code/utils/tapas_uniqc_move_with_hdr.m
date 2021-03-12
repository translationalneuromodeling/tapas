function [fileNameSourceArray, fileNameTargetArray, ...
    fileNameSourceMovedArray, fileNameTargetMovedArray] = tapas_uniqc_move_with_hdr(...
    fileNameSourceArray, fileNameTargetArray)
% Moves given files; for .nii (nifti-) files, also moves .mat-header,
% if existing; for given .img (analyze) files, also moves .hdr-header
%
%   [fileNameSourceArray, fileNameTargetArray] = tapas_uniqc_move_with_hdr(...
%    fileNameSourceArray, fileNameTargetArray)
%
% IN
%   fileNameSourceArray   cell of filenames to be moved (source)
%   fileNameTargetArray   cell of filenames to be moved to (targets)
% OUT
%   fileNameSourceArray   cell of filenames that were tried to be moved
%                   (includes   .mat files corresponding to .nii and
%                               .hdr files corresponding to .img)
%   fileNameSourceMovedArray   cell of source filenames that were actually moved
%                   (includes   .mat files corresponding to .nii and
%                               .hdr files corresponding to .img)
%   fileNameTargetMovedArray   cell of target filenames that were actually moved
%                   (includes   .mat files corresponding to .nii and
%                               .hdr files corresponding to .img)
%
% EXAMPLE
%   tapas_uniqc_move_with_hdr('from.nii', 'to.nii')
%
%   See also tapas_uniqc_delete_with_hdr copy_with_mat

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


doMove = true;
[fileNameSourceArray, fileNameTargetArray, ...
    fileNameSourceMovedArray, fileNameTargetMovedArray] = tapas_uniqc_copy_with_hdr(...
    fileNameSourceArray, fileNameTargetArray, doMove);