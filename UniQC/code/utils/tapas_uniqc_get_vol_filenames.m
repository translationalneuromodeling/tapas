% create file names of spm-analyze files for fMRI session w/ multiple vols
%
% INPUT
%   fn      - filename
%   selectedVolumes   - indices of all volumes to be considered
%             Inf for all volumes
%             (uses spm_select, only possible, if files already exist!)
%
% OUTPUT
%   nifti_flag - 1 to make nifti-compatible output, not analyze
%   fnames      [nVols, 1] cell array of filenames corresponding to specified volumes

function [fnames, nifti_flag] = tapas_uniqc_get_vol_filenames(fn, selectedVolumes)

[fp, fn, ext] = fileparts(fn);

nifti_flag = false;

switch ext
    case '.nii'
        nifti_flag = true;
    case '.hdr'
        ext = '.img';
end


% strip counter (e.g. '_0005') from analyze-header files belonging to a 4D dataset
if ~nifti_flag, fn = regexprep(fn, '_\d\d\d\d$', ''); end

doDetermineVolumes = nargin < 2 || any(isinf(selectedVolumes));

% if no volume indices are given, assume to take all volumes from
% correspondingly named files
if doDetermineVolumes
    
    fnames = cellstr(spm_select('ExtList', fp, ['^' fn, ext], Inf));
    
    % search for multiple volumes of .img
    if isempty(fnames{1}) && ~nifti_flag
        fnames = cellstr(spm_select('ExtList', fp, ['^' fn '_\d\d\d\d' ext], Inf));
    end
    
    % BUG: somehow on tnunash spm_select does not find data in Andreea's
    % folder
    if isempty(fnames{1})
        fnames = cellstr([fn, ext]);
    end
    
    % remove comma in .img-files
    if ~nifti_flag
        fnames = regexprep(fnames, ',.*', '');
    end
else
    cselectedVolumes = num2cell(selectedVolumes);
    if nifti_flag
        fnames = cellfun(@(x) sprintf('%s.nii,%d',fn,x),cselectedVolumes, 'UniformOutput', false);
    else
        fnames = cellfun(@(x) sprintf('%s_%04d.img',fn,x),cselectedVolumes, 'UniformOutput', false);
    end
end

if isempty(fnames{1})
    fnames = {};
else
    if ~isempty(fp)
        fnames = strcat(fp, filesep, fnames);
    end
end

fnames = reshape(fnames, [], 1);