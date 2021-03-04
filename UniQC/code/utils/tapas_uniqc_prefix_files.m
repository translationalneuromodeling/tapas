function pfxFileArray = tapas_uniqc_prefix_files(fileArray, pfx, isSuffix, isMixedCase)
%prefixes/or suffixes (cell of) files (incl. paths) with file prefix before file name
%
%   output = tapas_uniqc_prefix_files(input)
%
% IN
%   fileArray cell(nFiles,1) or single string of filenames
%       e.g.
%            '/LK215/functional/fMRI_session_1'
%             'LK215/functional/fMRI_session_2'
%   pfx         prefix or suffix, e.g. 'r' or '_GM'
%
%   isSuffix    if false (default), 'rfMRI_session_1.nii.' is created (incl. path)
%               if true, 'fMRI_session_1_GM.nii' is created
%   isMixedCase if false (default), pre/suffix added as is
%               e.g. prefifx_files('path/test.nii', 'new')
%                       -> path/newtest.nii
%               if true, mixed case conventions are obeyed,
%               prefifx_files('path/test.nii', 'new')
%                       -> path/newTest.nii
% OUT
%
% EXAMPLE
%   tapas_uniqc_prefix_files
%
%   See also

% Author: Lars Kasper
% Created: 2013-12-03
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.

if nargin < 3
    isSuffix = 0;
end

if nargin < 4
    isMixedCase = false;
end

if isMixedCase && isSuffix
    pfx(1) = upper(pfx(1));
end

isArray = iscell(fileArray);
if ~isArray
    fileArray = {fileArray};
end

nFiles = length(fileArray);

pfxFileArray = cell(nFiles,1);
for f = 1:nFiles
    fn = fileArray{f};
    [f1, f2, f3] = fileparts(fn);
    if isSuffix
        pfxFileArray{f} = fullfile(f1, [f2, pfx, f3]);
    else
        if isMixedCase
            f2(1) = upper(f2(1));
        end
        pfxFileArray{f} = fullfile(f1, [pfx, f2 , f3]);
    end
end

if ~isArray
    pfxFileArray = pfxFileArray{1};
end