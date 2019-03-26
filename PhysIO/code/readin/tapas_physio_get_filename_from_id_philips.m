function fn = tapas_physio_get_filename_from_id_philips(fileID, dirRaw, ext)
%searches for matching filename with fileID in dirRaw
%
%   output = get_filename_from_id(input)
%
% IN
%   fileID  e.g. 8
%   dirRaw  folder with raw files
%   ext     {'.nii'} file extension
% OUT
%
% EXAMPLE
%   get_filename_from_id
%
%   See also

% Author: Lars Kasper
% Created: 2013-11-04
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.

if nargin < 3
    ext = '.nii';
end

try
    fn = regexprep(regexprep(ls(fullfile(dirRaw ...
        ,['*_' num2str(fileID) '_1_*' ext])), [dirRaw filesep], ''),'\n','');
catch
    fn = regexprep(regexprep(ls(fullfile(dirRaw ...
        ,['*_' num2str(fileID) '_2_*' ext])), [dirRaw filesep], ''),'\n','');
end