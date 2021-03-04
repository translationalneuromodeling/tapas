function [dimLabels, dimValues, pfx, sfx] = tapas_uniqc_get_dim_labels_from_string(filename)
% parses file name (string) for patterns _<dimLabel><index>...
%
%  [dimLabels, dimValues, pfx, sfx] = tapas_uniqc_get_dim_labels_from_string(filename)

%
% IN
%   filename    string (e.g. file name) of the form
%               <prefix>_<dimLabel1><dimValue1>..._<dimLabelN><dimValueN>.<suffix>
%               e.g. image_sli035_echo001_asl000_dyn123_dif000.mat
% OUT
%   dimLabels   cell(1,nDims) of strings (
%               e.g. {'sli', 'echo', 'asl', 'dyn', 'dif'})
%   dimValues   [1, nDims] values for each dimension
%               e.g. [35, 1, 0, 123, 0]
%   pfx         (common) prefix of file 
%               (e.g. 'image')
%   sfx         suffix of file (e.g. .mat)
%
% EXAMPLE
%   tapas_uniqc_get_dim_labels_from_string
%
%   See also

% Author:   Lars Kasper
% Created:  2016-11-08
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% dim labels is everything that has no underscore and is followed by a
% number
% dimLabel/Value pairs are separated by underscore
[nonDimString, dimLabelValue] = regexp(filename, '_(?<label>[^0-9_])+(?<value>\d)+', 'split', 'names');

pfx = nonDimString{1};
sfx = '';
% consider non-dim parts of string only as separate suffix, if
% dimLabelValue pairs exist in between
if numel(nonDimString) > 1
    sfx = nonDimString{end};
end

dimLabels =  {dimLabelValue(:).label};
dimValues =  {dimLabelValue(:).value};
dimValues = cellfun(@str2num, dimValues);